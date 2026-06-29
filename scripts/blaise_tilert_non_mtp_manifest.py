#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a non-MTP Blaise-to-TileRT conversion manifest.

The manifest is a dry-run conversion contract. It verifies all source keys and
packed tensor shapes without importing torch, safetensors, or TileRT. It does
not write converted checkpoint shards.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

JsonDict = dict[str, object]
Shape = list[int]

DEFAULT_MODEL_DIR = Path(
    "/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft"
)


def _load_json(path: Path) -> JsonDict:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _weight_map(model_dir: Path) -> dict[str, str]:
    index = _load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no weight_map object")
    return {str(key): str(value) for key, value in weight_map.items()}


class HeaderReader:
    """Read safetensors metadata lazily, one shard header at a time."""

    def __init__(self, model_dir: Path, weight_map: dict[str, str]) -> None:
        self._model_dir = model_dir
        self._weight_map = weight_map
        self._headers: dict[str, JsonDict] = {}

    def meta(self, key: str) -> JsonDict | None:
        filename = self._weight_map.get(key)
        if filename is None:
            return None
        header = self._header(filename)
        meta = header.get(key)
        if not isinstance(meta, dict):
            return None
        return meta

    def _header(self, filename: str) -> JsonDict:
        cached = self._headers.get(filename)
        if cached is not None:
            return cached
        path = self._model_dir / filename
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header in {path}")
        self._headers[filename] = header
        return header


def _as_int(config: JsonDict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer config field {key}, got {value!r}")
    return value


def _as_float(config: JsonDict, key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric config field {key}, got {value!r}")
    return float(value)


def _direct_source(key: str, expected_shape: Shape | None = None) -> dict[str, object]:
    return {
        "key": key,
        "expected_shape": expected_shape,
        "role": "direct",
    }


def _nvfp4_sources(prefix: str, out_features: int, in_features: int) -> dict[str, object]:
    if in_features % 16 != 0:
        raise ValueError(f"{prefix}: NVFP4 expected input dimension divisible by 16")
    return {
        "sources": [
            _direct_source(f"{prefix}.weight_packed", [out_features, in_features // 2]),
            _direct_source(f"{prefix}.weight_scale", [out_features, in_features // 16]),
            _direct_source(f"{prefix}.weight_global_scale", [1]),
        ],
        "optional_sources": [
            _direct_source(f"{prefix}.input_global_scale", [1]),
        ],
    }


def _add_task(
    tasks: list[dict[str, object]],
    *,
    layer_idx: int | None,
    role: str,
    conversion: str,
    target: str,
    sources: list[dict[str, object]],
    optional_sources: list[dict[str, object]] | None = None,
    consumed_by_public_tilert: bool = True,
    notes: str = "",
) -> None:
    tasks.append(
        {
            "layer": layer_idx,
            "role": role,
            "conversion": conversion,
            "target": target,
            "sources": sources,
            "optional_sources": optional_sources or [],
            "consumed_by_public_tilert": consumed_by_public_tilert,
            "notes": notes,
        }
    )


def _add_attention_tasks(tasks: list[dict[str, object]], config: JsonDict, layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"
    hidden = _as_int(config, "hidden_size")
    q_lora = _as_int(config, "q_lora_rank")
    kv_lora = _as_int(config, "kv_lora_rank")
    qk_nope = _as_int(config, "qk_nope_head_dim")
    qk_rope = _as_int(config, "qk_rope_head_dim")
    v_head = _as_int(config, "v_head_dim")
    n_heads = _as_int(config, "num_attention_heads")
    index_heads = _as_int(config, "index_n_heads")
    index_head_dim = _as_int(config, "index_head_dim")

    direct_shapes = {
        "input_layernorm.weight": [hidden],
        "self_attn.q_a_layernorm.weight": [q_lora],
        "self_attn.kv_a_layernorm.weight": [kv_lora],
        "self_attn.indexer.k_norm.weight": [index_head_dim],
        "self_attn.indexer.k_norm.bias": [index_head_dim],
    }
    for suffix, shape in direct_shapes.items():
        _add_task(
            tasks,
            layer_idx=layer_idx,
            role=f"attention.{suffix}",
            conversion="direct_copy",
            target=f"{prefix}.{suffix}",
            sources=[_direct_source(f"{prefix}.{suffix}", shape)],
        )

    projections = {
        "self_attn.q_a_proj": (q_lora, hidden),
        "self_attn.q_b_proj": (n_heads * (qk_nope + qk_rope), q_lora),
        "self_attn.kv_a_proj_with_mqa": (kv_lora + qk_rope, hidden),
        "self_attn.kv_b_proj": (n_heads * (qk_nope + v_head), kv_lora),
        "self_attn.o_proj": (hidden, n_heads * v_head),
        "self_attn.indexer.wk": (index_head_dim, hidden),
        "self_attn.indexer.wq_b": (index_heads * index_head_dim, q_lora),
        "self_attn.indexer.weights_proj": (index_heads, hidden),
    }
    for suffix, (out_features, in_features) in projections.items():
        notes = ""
        if suffix == "self_attn.indexer.weights_proj":
            notes = (
                "TileRT public source treats this as bf16, but the Blaise checkpoint "
                "stores it as NVFP4 packed."
            )
        packed_sources = _nvfp4_sources(f"{prefix}.{suffix}", out_features, in_features)
        _add_task(
            tasks,
            layer_idx=layer_idx,
            role=f"attention.{suffix}",
            conversion="nvfp4_packed_projection",
            target=f"{prefix}.{suffix}.weight + TileRT scale/layout",
            sources=packed_sources["sources"],
            optional_sources=packed_sources["optional_sources"],
            notes=notes,
        )


def _add_dense_mlp_tasks(tasks: list[dict[str, object]], config: JsonDict, layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"
    hidden = _as_int(config, "hidden_size")
    intermediate = _as_int(config, "intermediate_size")
    _add_task(
        tasks,
        layer_idx=layer_idx,
        role="dense_mlp.post_attention_layernorm",
        conversion="direct_copy",
        target=f"{prefix}.post_attention_layernorm.weight",
        sources=[_direct_source(f"{prefix}.post_attention_layernorm.weight", [hidden])],
    )
    for suffix, out_features, in_features in (
        ("mlp.gate_proj", intermediate, hidden),
        ("mlp.up_proj", intermediate, hidden),
        ("mlp.down_proj", hidden, intermediate),
    ):
        packed_sources = _nvfp4_sources(f"{prefix}.{suffix}", out_features, in_features)
        _add_task(
            tasks,
            layer_idx=layer_idx,
            role=f"dense_mlp.{suffix}",
            conversion="nvfp4_packed_projection",
            target=f"{prefix}.{suffix}.weight + TileRT scale/layout",
            sources=packed_sources["sources"],
            optional_sources=packed_sources["optional_sources"],
        )


def _add_moe_tasks(tasks: list[dict[str, object]], config: JsonDict, layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"
    hidden = _as_int(config, "hidden_size")
    moe_intermediate = _as_int(config, "moe_intermediate_size")
    routed_experts = _as_int(config, "n_routed_experts")

    for suffix, shape in (
        ("post_attention_layernorm.weight", [hidden]),
        ("mlp.gate.weight", [routed_experts, hidden]),
        ("mlp.gate.e_score_correction_bias", [routed_experts]),
    ):
        _add_task(
            tasks,
            layer_idx=layer_idx,
            role=f"moe.{suffix}",
            conversion="direct_copy",
            target=f"{prefix}.{suffix}",
            sources=[_direct_source(f"{prefix}.{suffix}", shape)],
        )

    expert_prefixes = ["mlp.shared_experts"] + [
        f"mlp.experts.{expert_idx}" for expert_idx in range(routed_experts)
    ]
    for expert_prefix in expert_prefixes:
        for suffix, out_features, in_features in (
            ("gate_proj", moe_intermediate, hidden),
            ("up_proj", moe_intermediate, hidden),
            ("down_proj", hidden, moe_intermediate),
        ):
            source_prefix = f"{prefix}.{expert_prefix}.{suffix}"
            packed_sources = _nvfp4_sources(source_prefix, out_features, in_features)
            _add_task(
                tasks,
                layer_idx=layer_idx,
                role=f"moe.{expert_prefix}.{suffix}",
                conversion="nvfp4_packed_projection",
                target=f"{source_prefix}.weight + TileRT scale/layout",
                sources=packed_sources["sources"],
                optional_sources=packed_sources["optional_sources"],
            )


def _add_blaise_semantic_tasks(
    tasks: list[dict[str, object]], config: JsonDict, layer_idx: int
) -> None:
    prefix = f"model.layers.{layer_idx}"
    hidden = _as_int(config, "hidden_size")
    rank = _as_int(config, "gated_norm_rank")
    n_heads = _as_int(config, "num_attention_heads")
    v_head = _as_int(config, "v_head_dim")
    semantic_shapes = {
        "self_attn.gate_proj.weight": [n_heads * v_head, hidden],
        "input_gated_norm_down.weight": [rank, hidden],
        "input_gated_norm_up.weight": [hidden, rank],
        "post_attention_gated_norm_down.weight": [rank, hidden],
        "post_attention_gated_norm_up.weight": [hidden, rank],
    }
    for suffix, shape in semantic_shapes.items():
        _add_task(
            tasks,
            layer_idx=layer_idx,
            role=f"blaise_semantics.{suffix}",
            conversion="direct_copy_unconsumed_by_public_tilert",
            target=f"{prefix}.{suffix}",
            sources=[_direct_source(f"{prefix}.{suffix}", shape)],
            consumed_by_public_tilert=False,
            notes="Required for Blaise model semantics; not represented in TileRT public graph.",
        )


def _add_special_tasks(tasks: list[dict[str, object]], config: JsonDict) -> None:
    hidden = _as_int(config, "hidden_size")
    vocab = _as_int(config, "vocab_size")
    for key, shape in (
        ("model.embed_tokens.weight", [vocab, hidden]),
        ("model.norm.weight", [hidden]),
        ("lm_head.weight", [vocab, hidden]),
    ):
        _add_task(
            tasks,
            layer_idx=None,
            role=f"special.{key}",
            conversion="direct_copy",
            target=key,
            sources=[_direct_source(key, shape)],
        )


def _build_tasks(config: JsonDict) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    n_layers = _as_int(config, "num_hidden_layers")
    first_dense_replace = _as_int(config, "first_k_dense_replace")
    nextn = _as_int(config, "num_nextn_predict_layers")
    if nextn != 0:
        raise ValueError(f"This manifest is non-MTP only, got {nextn} NextN layers")

    _add_special_tasks(tasks, config)
    for layer_idx in range(n_layers):
        _add_attention_tasks(tasks, config, layer_idx)
        if layer_idx < first_dense_replace:
            _add_dense_mlp_tasks(tasks, config, layer_idx)
        else:
            _add_moe_tasks(tasks, config, layer_idx)
        _add_blaise_semantic_tasks(tasks, config, layer_idx)
    return tasks


def _validate_source(
    task: dict[str, object],
    source: dict[str, object],
    weight_map: dict[str, str],
    headers: HeaderReader,
    missing: list[dict[str, object]],
    shape_mismatches: list[dict[str, object]],
    require_presence: bool,
) -> None:
    key = str(source["key"])
    if key not in weight_map:
        if require_presence:
            missing.append({"task": task, "source": source})
        return
    expected_shape = source.get("expected_shape")
    if expected_shape is None:
        return
    meta = headers.meta(key)
    actual_shape = meta.get("shape") if meta is not None else None
    if actual_shape != expected_shape:
        shape_mismatches.append(
            {
                "task": task,
                "source": source,
                "actual_shape": actual_shape,
            }
        )


def _validate_tasks(
    tasks: list[dict[str, object]], weight_map: dict[str, str], headers: HeaderReader
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    missing: list[dict[str, object]] = []
    shape_mismatches: list[dict[str, object]] = []
    for task in tasks:
        sources = task["sources"]
        if not isinstance(sources, list):
            raise ValueError("Malformed manifest task: sources must be a list")
        optional_sources = task.get("optional_sources", [])
        if not isinstance(optional_sources, list):
            raise ValueError("Malformed manifest task: optional_sources must be a list")
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Malformed manifest source entry")
            _validate_source(
                task,
                source,
                weight_map,
                headers,
                missing,
                shape_mismatches,
                require_presence=True,
            )
        for source in optional_sources:
            if not isinstance(source, dict):
                raise ValueError("Malformed manifest optional source entry")
            _validate_source(
                task,
                source,
                weight_map,
                headers,
                missing,
                shape_mismatches,
                require_presence=False,
            )
    return missing, shape_mismatches


def _model_args_overrides(config: JsonDict) -> dict[str, object]:
    return {
        "arch_name": "deepseek_v3_2",
        "max_seq_len": _as_int(config, "max_position_embeddings"),
        "vocab_size": _as_int(config, "vocab_size"),
        "dim": _as_int(config, "hidden_size"),
        "inter_dim": _as_int(config, "intermediate_size"),
        "moe_inter_dim": _as_int(config, "moe_intermediate_size"),
        "n_layers": _as_int(config, "num_hidden_layers"),
        "n_dense_layers": _as_int(config, "first_k_dense_replace"),
        "n_heads": _as_int(config, "num_attention_heads"),
        "n_routed_experts": _as_int(config, "n_routed_experts"),
        "n_shared_experts": _as_int(config, "n_shared_experts"),
        "n_activated_experts": _as_int(config, "num_experts_per_tok"),
        "n_expert_groups": _as_int(config, "n_group"),
        "n_limited_groups": _as_int(config, "topk_group"),
        "score_func": config.get("scoring_func", "sigmoid"),
        "route_scale": _as_float(config, "routed_scaling_factor"),
        "q_lora_rank": _as_int(config, "q_lora_rank"),
        "kv_lora_rank": _as_int(config, "kv_lora_rank"),
        "qk_nope_head_dim": _as_int(config, "qk_nope_head_dim"),
        "qk_rope_head_dim": _as_int(config, "qk_rope_head_dim"),
        "v_head_dim": _as_int(config, "v_head_dim"),
        "index_n_heads": _as_int(config, "index_n_heads"),
        "index_head_dim": _as_int(config, "index_head_dim"),
        "index_topk": _as_int(config, "index_topk"),
        "block_size": 16,
        "num_mtp_layers": 0,
    }


def _task_required_sources(tasks: list[dict[str, object]]) -> set[str]:
    keys: set[str] = set()
    for task in tasks:
        sources = task["sources"]
        if not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, dict):
                keys.add(str(source["key"]))
    return keys


def _present_optional_sources(
    tasks: list[dict[str, object]], weight_map: dict[str, str]
) -> set[str]:
    keys: set[str] = set()
    for task in tasks:
        optional_sources = task.get("optional_sources", [])
        if not isinstance(optional_sources, list):
            continue
        for source in optional_sources:
            if not isinstance(source, dict):
                continue
            key = str(source["key"])
            if key in weight_map:
                keys.add(key)
    return keys


def _summarize_manifest(manifest: JsonDict) -> None:
    summary = manifest["summary"]
    if not isinstance(summary, dict):
        raise ValueError("Malformed manifest summary")
    print("# Blaise non-MTP TileRT manifest")
    for key in (
        "layers",
        "dense_layers",
        "moe_layers",
        "tasks",
        "unique_required_source_keys",
        "present_optional_source_keys",
        "missing_sources",
        "shape_mismatches",
        "public_tilert_unconsumed_semantic_tasks",
    ):
        print(f"- {key}: {summary[key]}")
    print("## Tasks by conversion")
    by_conversion = summary["tasks_by_conversion"]
    if isinstance(by_conversion, dict):
        for key, value in sorted(by_conversion.items()):
            print(f"- {key}: {value}")
    print("## ABI risks")
    abi_risks = manifest["abi_risks"]
    if isinstance(abi_risks, list):
        for risk in abi_risks:
            print(f"- {risk}")


def _first_items(items: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    return items[: max(limit, 0)]


def build_manifest(model_dir: Path) -> JsonDict:
    config = _load_json(model_dir / "config.json")
    weight_map = _weight_map(model_dir)
    headers = HeaderReader(model_dir, weight_map)
    tasks = _build_tasks(config)
    missing, shape_mismatches = _validate_tasks(tasks, weight_map, headers)

    tasks_by_conversion = Counter(str(task["conversion"]) for task in tasks)
    n_layers = _as_int(config, "num_hidden_layers")
    dense_layers = _as_int(config, "first_k_dense_replace")
    semantic_tasks = [
        task for task in tasks if task.get("consumed_by_public_tilert") is False
    ]
    required_source_keys = _task_required_sources(tasks)
    optional_source_keys = _present_optional_sources(tasks, weight_map)
    summary = {
        "layers": n_layers,
        "dense_layers": dense_layers,
        "moe_layers": n_layers - dense_layers,
        "tasks": len(tasks),
        "unique_required_source_keys": len(required_source_keys),
        "present_optional_source_keys": len(optional_source_keys),
        "missing_sources": len(missing),
        "shape_mismatches": len(shape_mismatches),
        "public_tilert_unconsumed_semantic_tasks": len(semantic_tasks),
        "tasks_by_conversion": dict(tasks_by_conversion),
    }
    return {
        "model_dir": str(model_dir),
        "model_args_overrides": _model_args_overrides(config),
        "summary": summary,
        "abi_risks": [
            "TileRT public binary may bake 256 routed experts; Blaise uses 128.",
            "TileRT public binary may bake index_topk=2048; Blaise uses 1024.",
            "TileRT public converter assumes one MTP layer; Blaise has no layer 61.",
            "TileRT public converter expects FP8 weights; Blaise stores NVFP4 packed weights.",
            "TileRT public graph omits Blaise attention output gate and GatedNorm semantics.",
        ],
        "missing_sources": missing,
        "shape_mismatches": shape_mismatches,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to the Blaise checkpoint directory.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Optional path for the full JSON manifest.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum missing/shape-mismatch entries to print in text mode.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.model_dir)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _summarize_manifest(manifest)
    missing = manifest["missing_sources"]
    if isinstance(missing, list) and missing:
        print("## Missing source examples")
        print(json.dumps(_first_items(missing, args.max_errors), indent=2))
    shape_mismatches = manifest["shape_mismatches"]
    if isinstance(shape_mismatches, list) and shape_mismatches:
        print("## Shape mismatch examples")
        print(json.dumps(_first_items(shape_mismatches, args.max_errors), indent=2))


if __name__ == "__main__":
    main()

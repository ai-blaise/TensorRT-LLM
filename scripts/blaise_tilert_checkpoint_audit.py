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

"""Audit a Blaise DeepSeek checkpoint against TileRT DeepSeek-V3.2 assumptions.

This intentionally uses only the Python standard library. It can run on a host
without ``torch``, ``safetensors``, or the TileRT wheel installed.
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
from pathlib import Path

JsonDict = dict[str, object]
ModelArgs = dict[str, object]


CONFIG_COMPARISONS = (
    ("dim", "hidden_size"),
    ("inter_dim", "intermediate_size"),
    ("moe_inter_dim", "moe_intermediate_size"),
    ("n_layers", "num_hidden_layers"),
    ("n_dense_layers", "first_k_dense_replace"),
    ("n_heads", "num_attention_heads"),
    ("n_routed_experts", "n_routed_experts"),
    ("n_shared_experts", "n_shared_experts"),
    ("n_activated_experts", "num_experts_per_tok"),
    ("n_expert_groups", "n_group"),
    ("n_limited_groups", "topk_group"),
    ("route_scale", "routed_scaling_factor"),
    ("q_lora_rank", "q_lora_rank"),
    ("kv_lora_rank", "kv_lora_rank"),
    ("qk_nope_head_dim", "qk_nope_head_dim"),
    ("qk_rope_head_dim", "qk_rope_head_dim"),
    ("v_head_dim", "v_head_dim"),
    ("index_n_heads", "index_n_heads"),
    ("index_head_dim", "index_head_dim"),
    ("index_topk", "index_topk"),
    ("max_seq_len", "max_position_embeddings"),
)

TILERT_FP8_SUFFIXES = (
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_proj.weight_scale_inv",
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.weight_scale_inv",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
    "self_attn.kv_b_proj.weight",
    "self_attn.kv_b_proj.weight_scale_inv",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.weight_scale_inv",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.wk.weight_scale_inv",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wq_b.weight_scale_inv",
    "self_attn.indexer.weights_proj.weight",
)

TILERT_DENSE_MLP_SUFFIXES = (
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight_scale_inv",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight_scale_inv",
    "mlp.down_proj.weight",
    "mlp.down_proj.weight_scale_inv",
)

BLAISE_SEMANTIC_SUFFIXES = (
    "self_attn.gate_proj.weight",
    "input_gated_norm_down.weight",
    "input_gated_norm_up.weight",
    "post_attention_gated_norm_down.weight",
    "post_attention_gated_norm_up.weight",
)

MTP_REF_SUFFIXES = (
    "enorm.weight",
    "hnorm.weight",
    "eh_proj.weight",
)

MTP_KEY_PATTERNS = (
    "model.layers.61",
    "mtp",
    "next",
    "draft",
    "spec",
    "eh_proj",
    "embedding_rmsnorm",
    "hidden_rmsnorm",
)


def _load_json(path: Path) -> JsonDict:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _eval_scalar(node: ast.AST) -> str | int | float | bool | None:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_scalar(node.left)
        right = _eval_scalar(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
    raise ValueError(f"Unsupported ModelArgs value: {ast.dump(node)}")


def _load_tilert_model_args(tilert_source: Path) -> ModelArgs:
    model_args_path = tilert_source / "tilert/models/deepseek_v3_2/model_args.py"
    module = ast.parse(
        model_args_path.read_text(encoding="utf-8"), filename=str(model_args_path)
    )
    values: ModelArgs = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ModelArgs":
            continue
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                target = item.targets[0]
                if isinstance(target, ast.Name):
                    values[target.id] = _eval_scalar(item.value)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                values[item.target.id] = _eval_scalar(item.value)
        return values
    raise ValueError(f"Could not find ModelArgs in {model_args_path}")


def _weight_map(model_dir: Path) -> dict[str, str]:
    index = _load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no weight_map object")
    return {str(key): str(value) for key, value in weight_map.items()}


def _safetensors_header(path: Path) -> JsonDict:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}")
    return header


def _tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> str:
    filename = weight_map.get(key)
    if filename is None:
        return "missing"
    header = _safetensors_header(model_dir / filename)
    meta = header.get(key)
    if not isinstance(meta, dict):
        return f"{filename}: header-missing"
    dtype = meta.get("dtype", "?")
    shape = meta.get("shape", "?")
    return f"{filename}: {dtype} {shape}"


def _layer_key_count(weight_map: dict[str, str], layer_idx: int) -> int:
    prefix = f"model.layers.{layer_idx}."
    return sum(1 for key in weight_map if key.startswith(prefix))


def _compressed_alternative(suffix: str) -> str:
    if suffix.endswith(".weight_scale_inv"):
        return suffix[: -len(".weight_scale_inv")] + ".weight_scale"
    if suffix.endswith(".weight"):
        return suffix[: -len(".weight")] + ".weight_packed"
    return suffix


def _print_config_comparison(config: JsonDict, tilert_args: ModelArgs) -> None:
    print("## Config comparison")
    print("| TileRT field | TileRT default | HF field | HF value | Status |")
    print("| --- | ---: | --- | ---: | --- |")
    for tilert_key, hf_key in CONFIG_COMPARISONS:
        tilert_value = tilert_args.get(tilert_key)
        hf_value = config.get(hf_key)
        status = "OK" if tilert_value == hf_value else "MISMATCH"
        print(f"| `{tilert_key}` | `{tilert_value}` | `{hf_key}` | `{hf_value}` | {status} |")


def _print_key_family(
    title: str,
    layer_idx: int,
    suffixes: tuple[str, ...],
    weight_map: dict[str, str],
) -> None:
    print(f"## {title}")
    missing = 0
    compressed = 0
    for suffix in suffixes:
        key = f"model.layers.{layer_idx}.{suffix}"
        if key in weight_map:
            continue
        alt_suffix = _compressed_alternative(suffix)
        alt_key = f"model.layers.{layer_idx}.{alt_suffix}"
        if alt_key in weight_map:
            compressed += 1
            continue
        missing += 1
        print(f"- Missing `{suffix}`")
    print(f"Expected suffixes: {len(suffixes)}")
    print(f"Present as TileRT-style keys: {len(suffixes) - missing - compressed}")
    print(f"Present only as compressed-tensors alternatives: {compressed}")
    print(f"Missing with no simple compressed alternative: {missing}")


def _print_blaise_semantics(layer_idx: int, weight_map: dict[str, str]) -> None:
    print("## Blaise-specific semantic tensors")
    for suffix in BLAISE_SEMANTIC_SUFFIXES:
        key = f"model.layers.{layer_idx}.{suffix}"
        status = "present" if key in weight_map else "missing"
        print(f"- `{suffix}`: {status}")


def _print_sample_tensor_shapes(model_dir: Path, weight_map: dict[str, str]) -> None:
    print("## Representative tensor storage")
    sample_keys = (
        "model.layers.0.self_attn.q_a_proj.weight_packed",
        "model.layers.0.self_attn.q_a_proj.weight_scale",
        "model.layers.0.self_attn.q_a_proj.weight_global_scale",
        "model.layers.0.self_attn.q_a_proj.input_global_scale",
        "model.layers.0.self_attn.gate_proj.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.layers.3.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.3.mlp.shared_experts.gate_proj.weight_packed",
        "lm_head.weight",
    )
    for key in sample_keys:
        print(f"- `{key}`: {_tensor_meta(model_dir, weight_map, key)}")


def _print_summary(config: JsonDict, tilert_args: ModelArgs, weight_map: dict[str, str]) -> None:
    print("## Summary")
    nextn_layers = config.get("num_nextn_predict_layers")
    layer_count = config.get("num_hidden_layers")
    tilert_total_layers = int(tilert_args.get("n_layers", 0) or 0) + 1
    layer_61_count = _layer_key_count(weight_map, 61)
    print(f"- HF `num_nextn_predict_layers`: `{nextn_layers}`")
    print(f"- HF base layer count: `{layer_count}`")
    print(f"- TileRT converter default total layers: `{tilert_total_layers}`")
    print(f"- Checkpoint `model.layers.61.*` keys: `{layer_61_count}`")
    print(f"- HF `attention_output_gate`: `{config.get('attention_output_gate')}`")
    print(f"- HF `gated_norm`: `{config.get('gated_norm')}`")
    print(f"- HF `quantization_scheme`: `{config.get('quantization_scheme')}`")


def _print_mtp_asset_audit(config: JsonDict, weight_map: dict[str, str]) -> None:
    print("## MTP asset audit")
    layer_count = int(config.get("num_hidden_layers", 0) or 0)
    nextn_layers = int(config.get("num_nextn_predict_layers", 0) or 0)
    mtp_prefix = f"model.layers.{layer_count}."
    mtp_layer_keys = [key for key in weight_map if key.startswith(mtp_prefix)]
    print(f"- HF `num_nextn_predict_layers`: `{nextn_layers}`")
    print(f"- Expected MTP layer prefix: `{mtp_prefix}*`")
    print(f"- MTP layer key count: `{len(mtp_layer_keys)}`")
    for suffix in MTP_REF_SUFFIXES:
        key = f"{mtp_prefix}{suffix}"
        status = "present" if key in weight_map else "missing"
        print(f"- `{key}`: {status}")
    for pattern in MTP_KEY_PATTERNS:
        hits = [key for key in weight_map if pattern.lower() in key.lower()]
        print(f"- Pattern `{pattern}` hits: `{len(hits)}`")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft"
        ),
        help="Path to the Blaise checkpoint directory.",
    )
    parser.add_argument(
        "--tilert-source",
        type=Path,
        default=Path("/home/sjpat/TileRT"),
        help="Path to a TileRT source checkout.",
    )
    args = parser.parse_args()

    config = _load_json(args.model_dir / "config.json")
    weight_map = _weight_map(args.model_dir)
    tilert_args = _load_tilert_model_args(args.tilert_source)

    print(f"# TileRT checkpoint audit: {args.model_dir}")
    _print_summary(config, tilert_args, weight_map)
    _print_mtp_asset_audit(config, weight_map)
    _print_config_comparison(config, tilert_args)
    _print_key_family("Layer 0 attention TileRT key check", 0, TILERT_FP8_SUFFIXES, weight_map)
    _print_key_family(
        "Layer 0 dense MLP TileRT key check",
        0,
        TILERT_DENSE_MLP_SUFFIXES,
        weight_map,
    )
    _print_key_family("Layer 3 attention TileRT key check", 3, TILERT_FP8_SUFFIXES, weight_map)
    _print_blaise_semantics(0, weight_map)
    _print_sample_tensor_shapes(args.model_dir, weight_map)


if __name__ == "__main__":
    main()

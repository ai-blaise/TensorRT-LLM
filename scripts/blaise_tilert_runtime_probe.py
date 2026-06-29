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

"""Probe the public TileRT runtime with Blaise DeepSeek-V3.2 shape overrides.

This is intentionally not a generation benchmark. It is the smallest runtime
gate before materializing hundreds of GiB of converted weights: load the TileRT
backend, build a Blaise-shaped ModelArgs object, and optionally construct DSA
scratch/cache tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path(
    "/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft"
)

MODEL_ARGS_FIELD_MAP = {
    "max_seq_len": "max_position_embeddings",
    "vocab_size": "vocab_size",
    "dim": "hidden_size",
    "inter_dim": "intermediate_size",
    "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers",
    "n_dense_layers": "first_k_dense_replace",
    "n_heads": "num_attention_heads",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "n_activated_experts": "num_experts_per_tok",
    "n_expert_groups": "n_group",
    "n_limited_groups": "topk_group",
    "route_scale": "routed_scaling_factor",
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "qk_nope_head_dim": "qk_nope_head_dim",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "v_head_dim": "v_head_dim",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
}

E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _parse_devices(raw_devices: str | None, visible_devices: int) -> list[int]:
    if raw_devices is None:
        return list(range(visible_devices))
    devices = [int(part.strip()) for part in raw_devices.split(",") if part.strip()]
    for device_id in devices:
        if device_id < 0 or device_id >= visible_devices:
            raise ValueError(
                f"Requested CUDA device {device_id}, but only {visible_devices} devices are visible"
            )
    return devices


def _build_model_args(args: argparse.Namespace) -> Any:
    from tilert.models.deepseek_v3_2.model_args import ModelArgs

    config = _load_json(args.model_dir / "config.json")
    model_args = ModelArgs()
    for model_arg_field, config_field in MODEL_ARGS_FIELD_MAP.items():
        setattr(model_args, model_arg_field, config[config_field])
    model_args.arch_name = "deepseek_v3_2"
    model_args.score_func = config.get("scoring_func", "sigmoid")
    model_args.block_size = args.block_size
    model_args.max_batch_size = args.max_batch_size
    model_args.kv_cache_pad = args.kv_cache_pad
    if args.max_seq_len is not None:
        model_args.max_seq_len = args.max_seq_len
    return model_args


def _model_args_summary(model_args: Any) -> dict[str, Any]:
    keys = (
        "arch_name",
        "max_batch_size",
        "max_seq_len",
        "dim",
        "n_layers",
        "n_dense_layers",
        "n_routed_experts",
        "n_activated_experts",
        "index_topk",
        "block_size",
        "score_func",
        "route_scale",
    )
    return {key: getattr(model_args, key) for key in keys}


def _shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile for an empty list")
    sorted_values = sorted(values)
    index = int(math.ceil(quantile * len(sorted_values))) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _load_backend() -> None:
    import tilert

    tilert.load_backend("deepseek_v3_2")


def _patch_flexible_tilert_weight_dequant() -> None:
    """Patch TileRT's reference dequant helper for partial 128-row groups.

    The public TileRT package assumes FP8 scale grids always describe complete
    128x128 blocks. DeepSeek-V3.2 has a 64-row RoPE slice and a 576-row KV-A
    combined projection, so the reference helper can fail before the backend
    path is reached. This is a probe-only patch, not a production loader.
    """

    import importlib
    import torch

    def flexible_weight_dequant(
        x_in: torch.Tensor, s_in: torch.Tensor, block_size: int = 128
    ) -> torch.Tensor:
        if not x_in.is_contiguous():
            x_in = x_in.contiguous()
        if not s_in.is_contiguous():
            s_in = s_in.contiguous()
        if x_in.dim() != 2 or s_in.dim() != 2:
            raise AssertionError("Input tensors must have 2 dimensions")

        rows, cols = x_in.shape
        scale_rows, scale_cols = s_in.shape
        if scale_rows != math.ceil(rows / block_size):
            raise AssertionError(
                "Scale rows do not match ceil-div row groups: "
                f"rows={rows}, scale_rows={scale_rows}, block_size={block_size}"
            )
        if scale_cols != math.ceil(cols / block_size):
            raise AssertionError(
                "Scale columns do not match ceil-div column groups: "
                f"cols={cols}, scale_cols={scale_cols}, block_size={block_size}"
            )

        out = torch.empty(rows, cols, dtype=torch.get_default_dtype(), device=x_in.device)
        x_float = x_in.float()
        s_float = s_in.float()
        for row_group in range(scale_rows):
            row_start = row_group * block_size
            row_end = min(row_start + block_size, rows)
            for col_group in range(scale_cols):
                col_start = col_group * block_size
                col_end = min(col_start + block_size, cols)
                out[row_start:row_end, col_start:col_end] = (
                    x_float[row_start:row_end, col_start:col_end]
                    * s_float[row_group, col_group]
                )
        return out

    module_names = [
        "tilert.models.common",
        "tilert.models.deepseek_v3_2.refs.kernel",
        "tilert.models.deepseek_v3_2.ops.down_allreduce",
        "tilert.models.deepseek_v3_2.ops.expert_down_allreduce",
        "tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu",
        "tilert.models.deepseek_v3_2.ops.projo_wkvb",
        "tilert.models.deepseek_v3_2.ops.projq_wqb",
        "tilert.models.deepseek_v3_2.ops.projx_wqkva",
        "tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqb",
        "tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqi",
        "tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqakis",
        "tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqkva",
        "tilert.models.deepseek_v3_2.ops.rmsnorm_up_gate_silu",
        "tilert.models.deepseek_v3_2.ops.unproj_o_allreduce",
    ]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "weight_dequant"):
            setattr(module, "weight_dequant", flexible_weight_dequant)


def _weight_map(model_dir: Path) -> dict[str, str]:
    index = _load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no weight_map object")
    return {str(key): str(value) for key, value in weight_map.items()}


def _load_layer_tensors(
    model_dir: Path, weight_map: dict[str, str], layer_idx: int, device: int
) -> dict[str, Any]:
    from safetensors import safe_open

    prefix = f"model.layers.{layer_idx}."
    layer_keys = sorted(key for key in weight_map if key.startswith(prefix))
    if not layer_keys:
        raise ValueError(f"No checkpoint tensors found for layer {layer_idx}")

    keys_by_file: dict[str, list[str]] = {}
    for key in layer_keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)

    tensors = {}
    for filename, keys in keys_by_file.items():
        with safe_open(model_dir / filename, framework="pt", device=f"cuda:{device}") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _dequant_nvfp4_to_bf16(
    packed: Any, scales: Any, global_scale: Any, block_size: int = 16
) -> Any:
    import torch

    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    dequant = torch.empty(
        packed.shape[0],
        packed.shape[1] * 2,
        dtype=torch.float32,
        device=packed.device,
    )
    dequant[:, 0::2] = table[low.long()]
    dequant[:, 1::2] = table[high.long()]
    scale_expanded = scales.float().repeat_interleave(block_size, dim=1)
    global_scale_inv = (1.0 / global_scale.float()).reshape(1, 1)
    return (dequant * scale_expanded * global_scale_inv).to(torch.bfloat16).contiguous()


def _quantize_fp8_blocks(weight: Any, block_size: int = 128) -> tuple[Any, Any]:
    import torch

    rows, cols = weight.shape
    scale_rows = math.ceil(rows / block_size)
    scale_cols = math.ceil(cols / block_size)
    quantized = torch.empty(
        rows,
        cols,
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    scales = torch.empty(scale_rows, scale_cols, dtype=torch.float32, device=weight.device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weight_float = weight.float()
    for row_group in range(scale_rows):
        row_start = row_group * block_size
        row_end = min(row_start + block_size, rows)
        for col_group in range(scale_cols):
            col_start = col_group * block_size
            col_end = min(col_start + block_size, cols)
            block = weight_float[row_start:row_end, col_start:col_end]
            scale = (block.abs().amax() / fp8_max).clamp(min=1e-12)
            scales[row_group, col_group] = scale
            quantized[row_start:row_end, col_start:col_end] = (
                (block / scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
            )
    return quantized.contiguous(), scales.contiguous()


def _add_tilert_fp8_aliases(layer_tensors: dict[str, Any]) -> dict[str, Any]:
    converted = dict(layer_tensors)
    packed_suffix = ".weight_packed"
    for packed_key in list(layer_tensors):
        if not packed_key.endswith(packed_suffix):
            continue
        base = packed_key[: -len(packed_suffix)]
        scale_key = f"{base}.weight_scale"
        global_scale_key = f"{base}.weight_global_scale"
        if scale_key not in layer_tensors or global_scale_key not in layer_tensors:
            raise KeyError(f"Missing NVFP4 scale tensors for {base}")
        weight_bf16 = _dequant_nvfp4_to_bf16(
            layer_tensors[packed_key],
            layer_tensors[scale_key],
            layer_tensors[global_scale_key],
        )
        if base.endswith("self_attn.indexer.weights_proj"):
            converted[f"{base}.weight"] = weight_bf16
        else:
            fp8_weight, fp8_scale = _quantize_fp8_blocks(weight_bf16)
            converted[f"{base}.weight"] = fp8_weight
            converted[f"{base}.weight_scale_inv"] = fp8_scale
        del weight_bf16
    return converted


def _pad_first_dim(tensor: Any, target_dim: int, fill_value: float = 0.0) -> Any:
    if target_dim <= tensor.shape[0]:
        return tensor
    padded_shape = (target_dim, *tensor.shape[1:])
    if fill_value == 0.0:
        padded = tensor.new_zeros(padded_shape)
    else:
        padded = tensor.new_full(padded_shape, fill_value)
    padded[: tensor.shape[0]].copy_(tensor)
    return padded.contiguous()


def _pad_moe_tilert_weights_for_native_abi(
    moe_weights: dict[str, dict[str, Any]],
    target_routed_experts: int,
) -> dict[str, dict[str, Any]]:
    """Pad converted Blaise MoE tensors to public DSV32 native expert slots."""

    if target_routed_experts <= 0:
        return moe_weights

    target_total_experts = target_routed_experts + 1
    padded_weights: dict[str, dict[str, Any]] = {}
    for dev_key, tensors in moe_weights.items():
        dev_tensors = dict(tensors)
        if "exp_bias" in dev_tensors:
            dev_tensors["exp_bias"] = _pad_first_dim(
                dev_tensors["exp_bias"],
                target_routed_experts,
                fill_value=-10000.0,
            )
        if "exp_proj_weights" in dev_tensors:
            dev_tensors["exp_proj_weights"] = _pad_first_dim(
                dev_tensors["exp_proj_weights"],
                target_routed_experts,
            )
        for key in (
            "exp_gate_weights",
            "exp_gate_scales",
            "exp_up_weights",
            "exp_up_scales",
            "exp_down_weights",
            "exp_down_scales",
        ):
            if key in dev_tensors:
                dev_tensors[key] = _pad_first_dim(dev_tensors[key], target_total_experts)
        padded_weights[dev_key] = dev_tensors
    return padded_weights


def _convert_layer_to_tilert_aliases(
    model_args: Any,
    args: argparse.Namespace,
    layer_idx: int,
    device: int,
) -> dict[str, dict[str, Any]]:
    from tilert.models.preprocess.weight_converter import WeightConverter

    weight_map = _weight_map(args.model_dir)
    layer_tensors = _load_layer_tensors(args.model_dir, weight_map, layer_idx, device)
    compat_tensors = _add_tilert_fp8_aliases(layer_tensors)
    converter = WeightConverter(
        model_args=model_args,
        num_devices=args.num_devices or 8,
        model_dir=str(args.model_dir),
        save_dir="/tmp/blaise-tilert-unused",
        test_mode=True,
    )
    mla_weights = converter.transform_mla(compat_tensors, layer_idx)
    if layer_idx < model_args.n_dense_layers:
        ffn_weights = converter.transform_mlp(compat_tensors, layer_idx)
    else:
        ffn_weights = converter.transform_moe(compat_tensors, layer_idx)
        ffn_weights = _pad_moe_tilert_weights_for_native_abi(
            ffn_weights,
            args.native_expert_weight_pad,
        )
    return {"mla": mla_weights, "ffn": ffn_weights}


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str, device: int) -> Any:
    from safetensors import safe_open

    with safe_open(model_dir / weight_map[key], framework="pt", device=f"cuda:{device}") as handle:
        return handle.get_tensor(key)


def _to_device_tensor(tensor: Any, device: int) -> Any:
    return tensor.to(device=f"cuda:{device}", non_blocking=True).contiguous()


def _stage_log(event: str, **fields: Any) -> None:
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    if field_text:
        print(f"[blaise-stream-step] {event} {field_text}", flush=True)
    else:
        print(f"[blaise-stream-step] {event}", flush=True)


def _sync_device(device_id: int, stage: str) -> None:
    import torch

    _stage_log("sync-begin", stage=stage, device=device_id)
    torch.cuda.synchronize(device_id)
    _stage_log(
        "sync-end",
        stage=stage,
        device=device_id,
        allocated_mib=torch.cuda.memory_allocated(device_id) // (1024 * 1024),
        reserved_mib=torch.cuda.memory_reserved(device_id) // (1024 * 1024),
    )


def _tensor_list_bytes(tensors: list[Any]) -> int:
    return sum(int(tensor.nbytes) for tensor in tensors if hasattr(tensor, "nbytes"))


def _pad_runtime_abi_buffers(
    args: argparse.Namespace,
    model_args: Any,
    device_id: int,
    dsa: Any,
    temp_vars: list[Any],
) -> None:
    """Over-allocate selected runtime buffers to probe baked native constants."""

    import torch
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    if args.native_index_topk_pad > model_args.index_topk:
        old_idx_selects = temp_vars[Idx.IDX_SELECTS]
        padded_idx_selects = torch.zeros(
            *old_idx_selects.shape[:-1],
            args.native_index_topk_pad,
            dtype=old_idx_selects.dtype,
            device=old_idx_selects.device,
        )
        padded_idx_selects[..., : old_idx_selects.shape[-1]].copy_(old_idx_selects)
        temp_vars[Idx.IDX_SELECTS] = padded_idx_selects
        _stage_log(
            "native-pad-idx-selects",
            device=device_id,
            old_shape=_shape(old_idx_selects),
            new_shape=_shape(padded_idx_selects),
        )

        if device_id != 0:
            old_ll_buf = dsa.v2_ll_buf
            new_ll_buf = torch.zeros(
                (getattr(model_args, "num_mtp", 3) + 1) * args.native_index_topk_pad * 2,
                dtype=old_ll_buf.dtype,
                device=old_ll_buf.device,
            )
            new_ll_buf[: old_ll_buf.numel()].copy_(old_ll_buf)
            dsa.v2_ll_buf = new_ll_buf
            for block in dsa.exec_seq:
                mla = getattr(block, "mla", None)
                if mla is not None and hasattr(mla, "ll_buf"):
                    mla.ll_buf = new_ll_buf
            _stage_log(
                "native-pad-ll-buf",
                device=device_id,
                old_shape=_shape(old_ll_buf),
                new_shape=_shape(new_ll_buf),
            )

    if args.native_routed_experts_pad > model_args.n_routed_experts:
        old_scores = temp_vars[Idx.SCORES]
        padded_scores = torch.zeros(
            *old_scores.shape[:-1],
            args.native_routed_experts_pad,
            dtype=old_scores.dtype,
            device=old_scores.device,
        )
        padded_scores[..., : old_scores.shape[-1]].copy_(old_scores)
        temp_vars[Idx.SCORES] = padded_scores
        _stage_log(
            "native-pad-scores",
            device=device_id,
            old_shape=_shape(old_scores),
            new_shape=_shape(padded_scores),
        )

    if args.native_idx_sel_ws_pad is not None:
        old_ws = temp_vars[Idx.IDX_SEL_WS]
        if args.native_idx_sel_ws_pad > old_ws.shape[-1]:
            padded_ws = torch.zeros(
                *old_ws.shape[:-1],
                args.native_idx_sel_ws_pad,
                dtype=old_ws.dtype,
                device=old_ws.device,
            )
            padded_ws[..., : old_ws.shape[-1]].copy_(old_ws)
            temp_vars[Idx.IDX_SEL_WS] = padded_ws
            _stage_log(
                "native-pad-idx-sel-ws",
                device=device_id,
                old_shape=_shape(old_ws),
                new_shape=_shape(padded_ws),
            )


def _force_native_partial_buf_shape(
    model_args: Any,
    device_id: int,
    dsa: Any,
    forward_seq_len: int,
) -> None:
    """Keep device-0 partial allreduce scratch batch-one for public native ABI."""

    if device_id != 0 or not hasattr(dsa, "v2_partial_buf"):
        return

    import torch

    old_partial = dsa.v2_partial_buf
    target_seq_len = max(forward_seq_len, old_partial.shape[1])
    if old_partial.shape[0] == 1 and old_partial.shape[1] >= forward_seq_len:
        return

    new_partial = torch.zeros(
        1,
        target_seq_len,
        model_args.dim,
        dtype=old_partial.dtype,
        device=old_partial.device,
    )
    copy_seq_len = min(old_partial.shape[1], new_partial.shape[1])
    new_partial[:, :copy_seq_len, :].copy_(old_partial[:1, :copy_seq_len, :])
    dsa.v2_partial_buf = new_partial
    for block in dsa.exec_seq:
        mla = getattr(block, "mla", None)
        if mla is not None and hasattr(mla, "partial_buf"):
            mla.partial_buf = new_partial
    _stage_log(
        "native-force-partial-buf",
        device=device_id,
        old_shape=_shape(old_partial),
        new_shape=_shape(new_partial),
    )


def probe_blaise_layer_init(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import tilert
    from tilert.models.deepseek_v3_2.modules.dsa import Dsa

    _load_backend()
    _patch_flexible_tilert_weight_dequant()
    visible_devices = torch.cuda.device_count()
    if args.target_device < 0 or args.target_device >= visible_devices:
        raise ValueError(
            f"Requested CUDA device {args.target_device}, "
            f"but only {visible_devices} devices are visible"
        )
    if args.block_size != 128:
        raise ValueError("blaise-layer-init requires --block-size 128 for TileRT FP8 compat")

    num_devices = args.num_devices or visible_devices
    model_args = _build_model_args(args)
    torch.cuda.set_device(args.target_device)
    before_allocated = torch.cuda.memory_allocated(args.target_device)
    before_reserved = torch.cuda.memory_reserved(args.target_device)

    converted = _convert_layer_to_tilert_aliases(
        model_args, args, args.layer_idx, args.target_device
    )
    dev_key = f"dev_{args.target_device}"
    block_state = {
        **converted["mla"][dev_key],
        **converted["ffn"][dev_key],
    }

    dsa = Dsa(model_args, device_id=args.target_device, num_devices=num_devices)
    block = dsa.exec_seq[args.layer_idx]
    block.init_tilert_weights(block_state)
    weights = block.get_weights_list()
    after_allocated = torch.cuda.memory_allocated(args.target_device)
    after_reserved = torch.cuda.memory_reserved(args.target_device)
    report = {
        "mode": "blaise-layer-init",
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": visible_devices,
        "num_devices_arg": num_devices,
        "target_device": args.target_device,
        "layer_idx": args.layer_idx,
        "model_args": _model_args_summary(model_args),
        "state_key_count": len(block_state),
        "weights_list_count": len(weights),
        "weights_bytes": sum(int(weight.nbytes) for weight in weights),
        "first_state_keys": sorted(block_state)[:20],
        "allocated_delta_bytes": after_allocated - before_allocated,
        "reserved_delta_bytes": after_reserved - before_reserved,
    }
    del weights
    del block
    del dsa
    del block_state
    del converted
    torch.cuda.empty_cache()
    return report


def _prepare_blaise_runtime(
    args: argparse.Namespace,
    model_args: Any,
    dsa_objects: list[Any],
    batch_size: int,
    forward_seq_len: int | None = None,
    mtp_objects: list[Any] | None = None,
) -> tuple[list[Any], float]:
    import torch
    from tilert.models.deepseek_v3_2.modules.end2end import dsa_show_hands_prepare_money
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx
    from tilert.utils import get_profile_log_tensor

    num_devices = len(dsa_objects)
    device_results = []
    base_counts = []
    p2p: dict[int, dict[str, Any]] = {}
    prep_start = time.perf_counter()
    seq_len = forward_seq_len or args.forward_seq_len
    _stage_log("prep-start", batch_size=batch_size, forward_seq_len=seq_len)
    for device_id, dsa in enumerate(dsa_objects):
        torch.cuda.set_device(device_id)
        _stage_log("prep-temp-vars-begin", device=device_id, batch_size=batch_size)
        temp_vars = dsa.get_temp_vars(
            batch_size=batch_size,
            seq_len=seq_len,
            extra_args={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_topp": args.use_topp,
            },
        )
        _sync_device(device_id, "prep-temp-vars")
        _stage_log(
            "prep-temp-vars-end",
            device=device_id,
            count=len(temp_vars),
            token_out_shape=_shape(temp_vars[Idx.TOKEN_OUT]),
        )
        _pad_runtime_abi_buffers(args, model_args, device_id, dsa, temp_vars)
        _force_native_partial_buf_shape(model_args, device_id, dsa, seq_len)

        _stage_log("prep-continuous-storage-begin", device=device_id)
        intermediates = _continuous_storage(temp_vars, device_id)
        _sync_device(device_id, "prep-continuous-storage")
        _stage_log(
            "prep-continuous-storage-end",
            device=device_id,
            count=len(intermediates),
            bytes=_tensor_list_bytes(intermediates),
        )

        _stage_log("prep-sampling-config-ready", device=device_id)

        _stage_log("prep-cache-vars-begin", device=device_id)
        caches = dsa.get_cache_vars()
        _sync_device(device_id, "prep-cache-vars")
        _stage_log(
            "prep-cache-vars-end",
            device=device_id,
            count=len(caches),
            first_shape=_shape(caches[0]) if caches else [],
            bytes=_tensor_list_bytes(caches),
        )

        _stage_log("prep-params-begin", device=device_id)
        params = dsa.get_weights_list()
        base_params_count = len(params)
        base_caches_count = len(caches)
        if mtp_objects is not None:
            mtp = mtp_objects[device_id]
            params.extend(mtp.get_weights_list())
            caches.extend(mtp.get_cache_vars())
        _stage_log(
            "prep-params-end",
            device=device_id,
            count=len(params),
            bytes=_tensor_list_bytes(params),
        )
        base_counts.append((base_params_count, base_caches_count))

        _stage_log("prep-profile-log-begin", device=device_id)
        profile_logs = get_profile_log_tensor(
            device=device_id,
            num_max_insts=args.profile_log_num_max_insts,
        )
        _sync_device(device_id, "prep-profile-log")
        _stage_log("prep-profile-log-end", device=device_id, shape=_shape(profile_logs))

        if device_id == 0:
            p2p[device_id] = {"peer_bufs": dsa.v2_peer_bufs}
            _stage_log(
                "prep-p2p-recorded",
                device=device_id,
                peer_bufs_ptr=int(dsa.v2_peer_bufs.data_ptr()),
            )
        else:
            p2p[device_id] = {"ll_buf": dsa.v2_ll_buf}
            _stage_log(
                "prep-p2p-recorded",
                device=device_id,
                ll_buf_ptr=int(dsa.v2_ll_buf.data_ptr()),
            )
        device_results.append((intermediates, caches, params, profile_logs))
        _stage_log("prep-device-end", device=device_id)

    _stage_log("prep-sampling-config-copy-begin", top_k=args.top_k)
    _set_sampling_config(args, device_results, args.top_k)
    _stage_log("prep-sampling-config-copy-end", top_k=args.top_k)

    _stage_log("p2p-peer-buffer-build-begin")
    peer_bufs_cpu = torch.zeros(num_devices - 1, dtype=torch.int64)
    for device_idx in range(num_devices - 1):
        peer_device = device_idx + 1
        peer_bufs_cpu[device_idx] = p2p[peer_device]["ll_buf"].data_ptr()
        _stage_log(
            "p2p-peer-buffer-entry",
            peer_device=peer_device,
            ptr=int(peer_bufs_cpu[device_idx].item()),
        )
    _stage_log("p2p-peer-buffer-copy-begin")
    p2p[0]["peer_bufs"].copy_(peer_bufs_cpu)
    _sync_device(0, "p2p-peer-buffer-copy")
    _stage_log("p2p-peer-buffer-copy-end")

    _stage_log("prepare-money-all-begin", batch_size=batch_size)
    for device_id, result in enumerate(device_results):
        torch.cuda.set_device(device_id)
        intermediates, caches, params, profile_logs = result
        _stage_log(
            "prepare-money-begin",
            device=device_id,
            params=len(params),
            intermediates=len(intermediates),
            caches=len(caches),
        )
        base_params_count, base_caches_count = base_counts[device_id]
        if mtp_objects is not None:
            dsa_show_hands_prepare_money(
                params,
                intermediates,
                caches,
                profile_logs,
                seq_len,
                True,
                False,
            )
            _sync_device(device_id, "prepare-money-mtp")
        dsa_show_hands_prepare_money(
            params[:base_params_count],
            intermediates,
            caches[:base_caches_count],
            profile_logs,
            seq_len,
            False,
            False,
        )
        _sync_device(device_id, "prepare-money")
        _stage_log("prepare-money-end", device=device_id)
    _stage_log("prepare-money-cross-sync-begin")
    for device_id in range(num_devices):
        _sync_device(device_id, "prepare-money-cross-sync")
    _stage_log("prepare-money-all-end", batch_size=batch_size, forward_seq_len=seq_len)
    prepare_seconds = time.perf_counter() - prep_start
    return device_results, prepare_seconds


def _make_decode_token(
    args: argparse.Namespace,
    batch_size: int,
    forward_seq_len: int,
    packed_concurrency: bool = False,
) -> Any:
    import torch

    if packed_concurrency:
        return torch.full((1, forward_seq_len), args.token_id, dtype=torch.long)
    if batch_size == 1:
        return torch.tensor(args.token_id, dtype=torch.long)
    return torch.full((batch_size,), args.token_id, dtype=torch.long)


def _memory_reports(num_devices: int) -> list[dict[str, int]]:
    import torch

    reports = []
    for device_id in range(num_devices):
        reports.append(
            {
                "device": device_id,
                "allocated_bytes": torch.cuda.memory_allocated(device_id),
                "reserved_bytes": torch.cuda.memory_reserved(device_id),
            }
        )
    return reports


def _sampling_top_ks(args: argparse.Namespace) -> list[int]:
    if args.sweep_top_ks:
        return _parse_int_list(args.sweep_top_ks)
    return [args.top_k]


def _runtime_max_seq_lens(args: argparse.Namespace, model_args: Any) -> list[int]:
    if args.sweep_max_seq_lens:
        return _parse_int_list(args.sweep_max_seq_lens)
    return [model_args.max_seq_len]


def _set_sampling_config(
    args: argparse.Namespace,
    device_results: list[Any],
    top_k: int,
) -> None:
    import torch
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    for device_id, result in enumerate(device_results):
        torch.cuda.set_device(device_id)
        intermediates = result[0]
        intermediates[Idx.SAMPLING_CONFIG].copy_(
            torch.tensor(
                [args.temperature, args.top_p, float(top_k), 1.0 if args.use_topp else 0.0],
                dtype=torch.float32,
                device=device_id,
            )
        )
    for device_id in range(len(device_results)):
        torch.cuda.synchronize(device_id)


def _reset_runtime_cache_tensors(module: Any) -> None:
    for attr_name in ("ki_cache", "kv_cache", "pe_cache"):
        if hasattr(module, attr_name):
            setattr(module, attr_name, None)
    for child in getattr(module, "exec_seq", []):
        _reset_runtime_cache_tensors(child)


def _set_runtime_max_seq_len(
    model_args: Any,
    dsa_objects: list[Any],
    max_seq_len: int,
) -> None:
    import torch

    model_args.max_seq_len = max_seq_len
    for device_id, dsa in enumerate(dsa_objects):
        torch.cuda.set_device(device_id)
        _reset_runtime_cache_tensors(dsa)
        dsa.freqs_cis = _freqs_cis(model_args, device_id)
    for device_id in range(len(dsa_objects)):
        torch.cuda.synchronize(device_id)
    torch.cuda.empty_cache()


def _profile_log_device_ids(args: argparse.Namespace, num_devices: int) -> list[int]:
    if not args.dump_profile_logs:
        return []
    if args.profile_log_devices == "device0":
        return [0]
    return list(range(num_devices))


def _clear_profile_logs(args: argparse.Namespace, device_results: list[Any]) -> None:
    import torch

    device_ids = _profile_log_device_ids(args, len(device_results))
    for device_id in device_ids:
        torch.cuda.set_device(device_id)
        profile_logs = device_results[device_id][3]
        profile_logs.zero_()
    for device_id in device_ids:
        torch.cuda.synchronize(device_id)


def _profile_log_rows(
    profile_logs: Any,
    row_indices: Any,
    row_counts: Any,
    values_per_row: int,
) -> list[dict[str, Any]]:
    if row_indices.numel() == 0:
        return []

    rows = []
    selected_rows = profile_logs.index_select(0, row_indices).detach().cpu()
    selected_counts = row_counts.index_select(0, row_indices).detach().cpu()
    row_numbers = row_indices.detach().cpu().tolist()
    for row_number, nonzero_count, row in zip(row_numbers, selected_counts.tolist(), selected_rows):
        nonzero_values = row.reshape(-1)
        nonzero_values = nonzero_values[nonzero_values != 0]
        if nonzero_values.numel() == 0:
            rows.append(
                {
                    "row": int(row_number),
                    "nonzero_values": int(nonzero_count),
                    "min": 0,
                    "max": 0,
                    "span": 0,
                    "first_values": [],
                }
            )
            continue

        min_value = int(nonzero_values.min().item())
        max_value = int(nonzero_values.max().item())
        rows.append(
            {
                "row": int(row_number),
                "nonzero_values": int(nonzero_count),
                "min": min_value,
                "max": max_value,
                "span": max_value - min_value,
                "first_values": [
                    int(value) for value in nonzero_values[:values_per_row].tolist()
                ],
            }
        )
    return rows


def _summarize_profile_log_tensor(args: argparse.Namespace, profile_logs: Any) -> dict[str, Any]:
    import torch

    max_rows = max(args.profile_log_max_rows, 0)
    values_per_row = max(args.profile_log_values_per_row, 0)
    row_counts = torch.count_nonzero(profile_logs, dim=(1, 2))
    nonzero_indices = torch.nonzero(row_counts, as_tuple=False).flatten()
    nonzero_row_count = int(nonzero_indices.numel())
    summary: dict[str, Any] = {
        "shape": _shape(profile_logs),
        "nonzero_row_count": nonzero_row_count,
    }
    if nonzero_row_count == 0:
        return summary

    summary["first_nonzero_row"] = int(nonzero_indices[0].item())
    summary["last_nonzero_row"] = int(nonzero_indices[-1].item())
    if max_rows == 0:
        return summary

    first_rows = nonzero_indices[: min(max_rows, nonzero_row_count)]
    top_count = min(max_rows, int(row_counts.numel()))
    top_counts, top_rows = torch.topk(row_counts, k=top_count)
    top_rows = top_rows[top_counts > 0]
    summary["first_rows"] = _profile_log_rows(
        profile_logs,
        first_rows,
        row_counts,
        values_per_row,
    )
    summary["top_count_rows"] = _profile_log_rows(
        profile_logs,
        top_rows,
        row_counts,
        values_per_row,
    )
    return summary


def _summarize_profile_logs(
    args: argparse.Namespace,
    device_results: list[Any],
) -> dict[str, Any]:
    import torch

    summaries = {}
    for device_id in _profile_log_device_ids(args, len(device_results)):
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize(device_id)
        profile_logs = device_results[device_id][3]
        summaries[str(device_id)] = _summarize_profile_log_tensor(args, profile_logs)
    return summaries


def _measurement_semantics(
    serial_context_sweep: bool,
    packed_concurrency_sweep: bool,
    seqlen_sweep: bool,
    maxseq_sweep: bool,
    mtp_capacity_sweep: bool,
    sweep_value: int,
) -> dict[str, Any]:
    if mtp_capacity_sweep:
        return {
            "measurement_semantics": "mtp_capacity_graft",
            "serving_concurrency_valid": True,
            "measurement_note": (
                "Single public TileRT request using a grafted, untrained MTP layer; "
                "quality is not validated."
            ),
        }
    if maxseq_sweep:
        return {
            "measurement_semantics": "native_single_request_maxseq_sweep",
            "serving_concurrency_valid": True,
            "measurement_note": "Single public TileRT decode slot with varied runtime max_seq_len.",
        }
    if serial_context_sweep:
        if sweep_value == 1:
            return {
                "measurement_semantics": "native_single_request",
                "serving_concurrency_valid": True,
                "measurement_note": "Single public TileRT decode slot.",
            }
        return {
            "measurement_semantics": "serial_cache_swap_debug",
            "serving_concurrency_valid": False,
            "measurement_note": (
                "Logical requests are serialized through one public TileRT decode slot "
                "with Python cache-state copies."
            ),
        }
    if packed_concurrency_sweep:
        return {
            "measurement_semantics": "packed_sequence_lanes",
            "serving_concurrency_valid": False,
            "measurement_note": (
                "The public ABI accepts seq_len lanes here, not independent request slots."
            ),
        }
    if seqlen_sweep:
        return {
            "measurement_semantics": "native_sequence_lanes",
            "serving_concurrency_valid": False,
            "measurement_note": (
                "Tokens are multiple positions for one native request, not serving concurrency."
            ),
        }
    return {
        "measurement_semantics": "native_batch",
        "serving_concurrency_valid": True,
        "measurement_note": "Native batch path.",
    }


def _forward_sync_device_ids(args: argparse.Namespace, num_devices: int) -> list[int]:
    if args.forward_sync == "none":
        return []
    if args.forward_sync == "device0":
        return [0]
    return list(range(num_devices))


def _sync_forward(args: argparse.Namespace, num_devices: int) -> None:
    import torch

    for device_id in _forward_sync_device_ids(args, num_devices):
        torch.cuda.synchronize(device_id)


def _record_forward_events(
    args: argparse.Namespace,
    num_devices: int,
) -> tuple[list[Any], list[Any], list[int]]:
    import torch

    if not args.measure_cuda_events:
        return [], [], []

    if args.cuda_event_devices == "device0":
        device_ids = [0]
    else:
        device_ids = list(range(num_devices))
    start_events = []
    end_events = []
    for device_id in device_ids:
        torch.cuda.set_device(device_id)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        start_events.append(start)
        end_events.append(end)
    return start_events, end_events, device_ids


def _finish_forward_events(
    start_events: list[Any],
    end_events: list[Any],
    device_ids: list[int],
) -> dict[str, float]:
    import torch

    event_times = {}
    for device_id, end in zip(device_ids, end_events):
        torch.cuda.set_device(device_id)
        end.record()
    for device_id, start, end in zip(device_ids, start_events, end_events):
        end.synchronize()
        event_times[str(device_id)] = start.elapsed_time(end) / 1000.0
    return event_times


def _cuda_profiler_start() -> None:
    import torch

    result = torch.cuda.cudart().cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with code {result}")


def _cuda_profiler_stop() -> None:
    import torch

    result = torch.cuda.cudart().cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with code {result}")


def _clone_cache_states(
    device_results: list[Any],
    concurrency: int,
) -> list[list[list[Any]]]:
    """Create independent logical-request KV/cache snapshots.

    The public TileRT DSV3.2 native ABI is batch-one. This debug helper keeps
    native cache tensor pointers stable and swaps request-local cache contents
    through those pointers between serial native forwards.
    """

    states = []
    for _ in range(concurrency):
        request_state = []
        for _, caches, _, _ in device_results:
            request_state.append([cache.clone() for cache in caches])
        states.append(request_state)
    return states


def _copy_cache_state(dst_caches: list[Any], src_caches: list[Any]) -> None:
    for dst_cache, src_cache in zip(dst_caches, src_caches):
        dst_cache.copy_(src_cache)


def _copy_cache_state_prefix(
    dst_caches: list[Any],
    src_caches: list[Any],
    prefix_len: int,
) -> None:
    if prefix_len <= 0:
        return
    for dst_cache, src_cache in zip(dst_caches, src_caches):
        if dst_cache.dim() >= 2:
            dst_cache[:, :prefix_len].copy_(src_cache[:, :prefix_len])
        else:
            dst_cache.copy_(src_cache)


def _copy_cache_state_token(
    dst_caches: list[Any],
    src_caches: list[Any],
    token_pos: int,
) -> None:
    for dst_cache, src_cache in zip(dst_caches, src_caches):
        if dst_cache.dim() >= 2 and token_pos < dst_cache.shape[1]:
            dst_cache[:, token_pos : token_pos + 1].copy_(
                src_cache[:, token_pos : token_pos + 1]
            )
        else:
            dst_cache.copy_(src_cache)


def _cache_state_bytes(device_results: list[Any]) -> int:
    return sum(_tensor_list_bytes(caches) for _, caches, _, _ in device_results)


def _run_serial_context_round(
    args: argparse.Namespace,
    device_results: list[Any],
    request_cache_states: list[list[list[Any]]] | None,
    request_cur_positions: list[int],
    request_tokens: list[Any],
) -> list[int]:
    import torch
    from tilert.models.deepseek_v3_2.modules.end2end import dsa_show_hands
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    token_out_values = []
    num_devices = len(device_results)
    for request_idx, token in enumerate(request_tokens):
        cur_pos = request_cur_positions[request_idx]
        if request_cache_states is not None:
            for device_id, result in enumerate(device_results):
                torch.cuda.set_device(device_id)
                _, native_caches, _, _ = result
                _copy_cache_state_prefix(
                    native_caches,
                    request_cache_states[request_idx][device_id],
                    cur_pos,
                )
            for device_id in range(num_devices):
                torch.cuda.synchronize(device_id)

        if request_cache_states is not None or args.serial_context_set_cur_pos:
            torch.ops.tilert.dsa_show_hands_set_cur_pos(cur_pos)
        dsa_show_hands(token, False, False)
        _sync_forward(args, num_devices)

        if request_cache_states is not None:
            for device_id, result in enumerate(device_results):
                torch.cuda.set_device(device_id)
                _, native_caches, _, _ = result
                _copy_cache_state_token(
                    request_cache_states[request_idx][device_id],
                    native_caches,
                    cur_pos,
                )
            for device_id in range(num_devices):
                torch.cuda.synchronize(device_id)

        token_out = int(device_results[0][0][Idx.TOKEN_OUT][0, 0, 0].item())
        token_out_values.append(token_out)
        request_cur_positions[request_idx] += 1
        if args.serial_context_feed_output:
            request_tokens[request_idx] = torch.tensor(token_out, dtype=torch.long)
    return token_out_values


def _prefix_tilert_state(
    tensors: dict[str, Any],
    prefix: str,
    suffix: str,
    device_id: int,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}{suffix}": _to_device_tensor(tensor, device_id)
        for key, tensor in tensors.items()
    }


def _build_grafted_mtp_objects(
    args: argparse.Namespace,
    model_args: Any,
    converter: Any,
    weight_map: dict[str, str],
    dsa_objects: list[Any],
    norm_shards: Any,
    head_shards: Any,
) -> tuple[list[Any], float]:
    import torch
    from tilert.models.deepseek_v3_2.modules.mla_v2 import PureMlaV2, SparseSelectMlaV2
    from tilert.models.deepseek_v3_2.modules.mtp import MTP

    start = time.perf_counter()
    num_devices = len(dsa_objects)
    graft_layer = args.mtp_graft_layer
    if graft_layer < model_args.n_dense_layers or graft_layer >= model_args.n_layers:
        raise ValueError(
            "--mtp-graft-layer must be one of the MoE layers "
            f"[{model_args.n_dense_layers}, {model_args.n_layers - 1}], got {graft_layer}"
        )

    _stage_log("mtp-graft-load-begin", graft_layer=graft_layer)
    torch.cuda.set_device(args.conversion_device)
    layer_tensors = _load_layer_tensors(args.model_dir, weight_map, graft_layer, args.conversion_device)
    compat_tensors = _add_tilert_fp8_aliases(layer_tensors)
    mla_weights = converter.transform_mla(compat_tensors, graft_layer)
    ffn_weights = converter.transform_moe(compat_tensors, graft_layer)
    ffn_weights = _pad_moe_tilert_weights_for_native_abi(
        ffn_weights,
        args.native_expert_weight_pad,
    )
    del compat_tensors
    del layer_tensors

    mtp_objects = []
    prefix = f"layer_{model_args.n_layers}_"
    for device_id, dsa in enumerate(dsa_objects):
        torch.cuda.set_device(device_id)
        mtp_kwargs: dict[str, Any] = {}
        mtp_kwargs["mla_cls"] = SparseSelectMlaV2 if device_id == 0 else PureMlaV2
        mtp_kwargs["mla_num_devices"] = 1 if device_id == 0 else num_devices - 1
        if device_id == 0:
            mtp_kwargs["mla_kwargs"] = {"peer_bufs": dsa.v2_peer_bufs}
        else:
            mtp_kwargs["mla_kwargs"] = {"ll_buf": dsa.v2_ll_buf}
        mtp = MTP(model_args, device_id, num_devices, **mtp_kwargs)

        suffix = f"_dev_{device_id}"
        dev_key = f"dev_{device_id}"
        mtp_state = {
            "model.embed_tokens.weight": dsa.embed_tokens_weight,
            "freqs_cis": dsa.freqs_cis,
            f"{prefix}embedding_rmsnorm_gamma{suffix}": _to_device_tensor(
                norm_shards[device_id],
                device_id,
            ),
            f"{prefix}hidden_rmsnorm_gamma{suffix}": _to_device_tensor(
                norm_shards[device_id],
                device_id,
            ),
            f"{prefix}eh_proj_weights{suffix}": torch.zeros(
                model_args.dim,
                model_args.dim * 2 // num_devices,
                dtype=torch.bfloat16,
                device=f"cuda:{device_id}",
            ),
            f"{prefix}model.norm.weight{suffix}": _to_device_tensor(
                norm_shards[device_id],
                device_id,
            ),
            f"{prefix}lm_head.weight{suffix}": _to_device_tensor(
                head_shards[device_id],
                device_id,
            ),
        }
        mtp_state.update(
            _prefix_tilert_state(mla_weights[dev_key], prefix, suffix, device_id)
        )
        mtp_state.update(
            _prefix_tilert_state(ffn_weights[dev_key], prefix, suffix, device_id)
        )
        mtp.init_tilert_weights(mtp_state)
        mtp_objects.append(mtp)
        _sync_device(device_id, "mtp-graft-init")
        _stage_log(
            "mtp-graft-device-end",
            device=device_id,
            graft_layer=graft_layer,
            state_keys=len(mtp_state),
            weight_bytes=_tensor_list_bytes(mtp.get_weights_list()),
            cache_count=len(mtp.get_cache_vars()),
        )
        del mtp_state

    del mla_weights
    del ffn_weights
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - start
    _stage_log("mtp-graft-load-end", graft_layer=graft_layer, seconds=f"{elapsed:.2f}")
    return mtp_objects, elapsed


def _set_mtp_decode_state(
    args: argparse.Namespace,
    device_results: list[Any],
    cur_pos: int,
) -> None:
    import torch
    from tilert.models.deepseek_v3_2.modules.end2end import (
        dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token,
        dsa_mtp_e2e_show_hands_set_prefill_valid_tokens,
    )
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    for device_id, result in enumerate(device_results):
        torch.cuda.set_device(device_id)
        intermediates = result[0]
        intermediates[Idx.CUR_POS].fill_(cur_pos)
        intermediates[Idx.LAST_HIDDEN_STATES].zero_()
    for device_id in range(len(device_results)):
        torch.cuda.synchronize(device_id)
    dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(0, False)
    dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token(args.token_id, False)


def _set_mtp_cur_pos(device_results: list[Any], cur_pos: int) -> None:
    import torch
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    for device_id, result in enumerate(device_results):
        torch.cuda.set_device(device_id)
        result[0][Idx.CUR_POS].fill_(cur_pos)
    for device_id in range(len(device_results)):
        torch.cuda.synchronize(device_id)


def _make_mtp_draft_tokens(args: argparse.Namespace) -> Any:
    import torch

    return torch.full(
        (1, args.mtp_seq_len),
        args.token_id,
        dtype=torch.int32,
    )


def _make_mtp_draft_tokens_with_valid_count(
    args: argparse.Namespace,
    valid_count: int,
) -> Any:
    import torch

    if valid_count <= 0:
        raise ValueError("valid_count must be positive")
    if valid_count > args.mtp_seq_len:
        raise ValueError(
            f"valid_count {valid_count} exceeds mtp_seq_len {args.mtp_seq_len}"
        )
    tokens = torch.full((args.mtp_seq_len,), args.token_id, dtype=torch.int32)
    return tokens.reshape(1, args.mtp_seq_len)


def _run_mtp_prefill(
    args: argparse.Namespace,
    device_results: list[Any],
) -> tuple[list[float], list[int]]:
    import time
    from tilert.models.deepseek_v3_2.modules.end2end import (
        dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token,
        dsa_mtp_e2e_show_hands_set_prefill_valid_tokens,
    )

    prefill_len = args.mtp_prefill_len
    if prefill_len <= 1:
        return [], []
    if args.max_seq_len is not None and prefill_len > args.max_seq_len:
        raise ValueError(
            f"mtp_prefill_len {prefill_len} exceeds max_seq_len {args.max_seq_len}"
        )

    step_times = []
    valid_counts = []
    cur_pos = 0
    while cur_pos < prefill_len - 1:
        draft_end = min(cur_pos + args.mtp_seq_len, prefill_len)
        valid_count = draft_end - cur_pos
        draft_tokens = _make_mtp_draft_tokens_with_valid_count(args, valid_count)

        extra_pos = cur_pos + args.mtp_seq_len
        extra_token = args.token_id if extra_pos < prefill_len else args.token_id
        dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token(extra_token, False)
        dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(valid_count, False)

        start = time.perf_counter()
        _run_mtp_capacity_forward(args, device_results, draft_tokens)
        step_times.append(time.perf_counter() - start)
        valid_counts.append(valid_count)
        cur_pos += valid_count

    _set_mtp_cur_pos(device_results, prefill_len - 1)
    dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(0, False)
    dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token(args.token_id, False)
    return step_times, valid_counts


def _run_mtp_capacity_forward(
    args: argparse.Namespace,
    device_results: list[Any],
    draft_tokens: Any,
) -> tuple[int, list[int], Any]:
    import torch
    from tilert.models.deepseek_v3_2.modules.end2end import dsa_show_hands
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    dsa_show_hands(draft_tokens, True, False)
    _sync_forward(args, len(device_results))
    intermediates = device_results[0][0]
    accepted = int(intermediates[Idx.ACCEPTED_TOKENS][0].item())
    predicted = [int(value) for value in intermediates[Idx.PREDICTED_TOKENS][0].flatten().tolist()]
    next_draft = intermediates[Idx.NEXT_DRAFT_TOKENS].detach().cpu().to(torch.int32)
    if args.mtp_feed_next_draft:
        draft_tokens = next_draft.reshape(1, args.mtp_seq_len)
    return accepted, predicted[: args.mtp_seq_len], draft_tokens


def probe_blaise_stream_sweep(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import tilert
    from tilert.models.deepseek_v3_2.modules.dsa import Dsa
    from tilert.models.deepseek_v3_2.modules.end2end import (
        dsa_show_hands,
        dsa_show_hands_go_home,
        dsa_show_hands_reset,
        dsa_show_hands_set_sampling_seed,
    )
    from tilert.models.deepseek_v3_2.ops.rmsnorm_head_proj import RMSNormHeadProj
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx
    from tilert.models.preprocess.weight_converter import WeightConverter
    from tilert.tilert_init import tilert_init

    _load_backend()
    _patch_flexible_tilert_weight_dequant()
    visible_devices = torch.cuda.device_count()
    num_devices = args.num_devices or visible_devices
    if visible_devices < num_devices:
        raise ValueError(f"Need {num_devices} visible CUDA devices, got {visible_devices}")
    if args.block_size != 128:
        raise ValueError(f"{args.mode} requires --block-size 128 for TileRT FP8 compat")

    seqlen_sweep = args.mode == "blaise-stream-seqlen-sweep"
    packed_concurrency_sweep = args.mode == "blaise-stream-packed-concurrency-sweep"
    serial_context_sweep = args.mode == "blaise-stream-serial-context-sweep"
    maxseq_sweep = args.mode == "blaise-stream-maxseq-sweep"
    mtp_capacity_sweep = args.mode == "blaise-stream-mtp-capacity"
    if mtp_capacity_sweep:
        sweep_values = [1]
        if args.max_batch_size != 1:
            raise ValueError("blaise-stream-mtp-capacity requires --max-batch-size 1")
        if args.mtp_seq_len != 4:
            raise ValueError("The public TileRT MTP e2e path currently expects --mtp-seq-len 4")
        if args.mtp_prefill_len < 0:
            raise ValueError("--mtp-prefill-len must be non-negative")
        if args.mtp_prefill_len > 1 and args.mtp_start_pos != 0:
            raise ValueError("--mtp-prefill-len requires --mtp-start-pos 0")
    elif maxseq_sweep:
        sweep_values = _parse_int_list(args.sweep_max_seq_lens)
        if args.max_batch_size != 1:
            raise ValueError("blaise-stream-maxseq-sweep requires --max-batch-size 1")
        if args.forward_seq_len != 1:
            raise ValueError("blaise-stream-maxseq-sweep requires --forward-seq-len 1")
    elif seqlen_sweep:
        sweep_values = _parse_int_list(args.sweep_seq_lens)
        unsupported_seq_lens = [value for value in sweep_values if value not in (1, 2, 4)]
        if unsupported_seq_lens:
            raise ValueError(
                "blaise-stream-seqlen-sweep supports only forward sequence "
                f"lengths 1, 2, and 4, got {unsupported_seq_lens}"
            )
        if args.max_batch_size != 1:
            raise ValueError("blaise-stream-seqlen-sweep requires --max-batch-size 1")
    elif packed_concurrency_sweep:
        sweep_values = _parse_int_list(args.sweep_concurrencies)
        unsupported_concurrencies = [value for value in sweep_values if value not in (1, 2, 4)]
        if unsupported_concurrencies:
            raise ValueError(
                "blaise-stream-packed-concurrency-sweep supports only logical "
                f"concurrency 1, 2, and 4, got {unsupported_concurrencies}"
            )
        max_concurrency = max(sweep_values)
        if max_concurrency > args.max_batch_size:
            raise ValueError(
                f"Max packed concurrency {max_concurrency} exceeds --max-batch-size "
                f"{args.max_batch_size}"
            )
    elif serial_context_sweep:
        sweep_values = _parse_int_list(args.sweep_concurrencies)
        unsupported_concurrencies = [value for value in sweep_values if value not in (1, 2, 4)]
        if unsupported_concurrencies:
            raise ValueError(
                "blaise-stream-serial-context-sweep supports logical "
                f"concurrency 1, 2, and 4, got {unsupported_concurrencies}"
            )
        if args.max_batch_size != 1:
            raise ValueError(
                "blaise-stream-serial-context-sweep requires --max-batch-size 1 "
                "because the public native ABI accepts only batch-one cache tensors"
            )
        if args.forward_seq_len != 1:
            raise ValueError(
                "blaise-stream-serial-context-sweep requires --forward-seq-len 1; "
                "logical concurrency is represented by separate swapped cache states"
            )
    else:
        sweep_values = _parse_int_list(args.sweep_concurrencies)
        max_concurrency = max(sweep_values)
        if max_concurrency > args.max_batch_size:
            raise ValueError(
                f"Max sweep concurrency {max_concurrency} exceeds --max-batch-size "
                f"{args.max_batch_size}"
            )

    model_args = _build_model_args(args)
    if maxseq_sweep:
        model_args.max_seq_len = max(sweep_values)
    conversion_device = args.conversion_device
    if conversion_device < 0 or conversion_device >= visible_devices:
        raise ValueError(f"Invalid conversion device {conversion_device}")

    _stage_log("tilert-init-begin")
    tilert_init()
    for device_id in range(num_devices):
        _sync_device(device_id, "tilert-init")
    _stage_log("tilert-init-end")

    weight_map = _weight_map(args.model_dir)
    converter = WeightConverter(
        model_args=model_args,
        num_devices=num_devices,
        model_dir=str(args.model_dir),
        save_dir="/tmp/blaise-tilert-unused",
        test_mode=True,
    )

    start = time.perf_counter()
    dsa_objects = []
    for device_id in range(num_devices):
        torch.cuda.set_device(device_id)
        dsa_objects.append(Dsa(model_args, device_id=device_id, num_devices=num_devices))

    layer_reports = []
    for layer_idx in range(model_args.n_layers):
        layer_start = time.perf_counter()
        torch.cuda.set_device(conversion_device)
        layer_tensors = _load_layer_tensors(
            args.model_dir, weight_map, layer_idx, conversion_device
        )
        compat_tensors = _add_tilert_fp8_aliases(layer_tensors)
        mla_weights = converter.transform_mla(compat_tensors, layer_idx)
        if layer_idx < model_args.n_dense_layers:
            ffn_weights = converter.transform_mlp(compat_tensors, layer_idx)
        else:
            ffn_weights = converter.transform_moe(compat_tensors, layer_idx)
            ffn_weights = _pad_moe_tilert_weights_for_native_abi(
                ffn_weights,
                args.native_expert_weight_pad,
            )

        layer_weight_bytes = 0
        for device_id in range(num_devices):
            dev_key = f"dev_{device_id}"
            block_state = {}
            for source in (mla_weights[dev_key], ffn_weights[dev_key]):
                for key, tensor in source.items():
                    block_state[key] = _to_device_tensor(tensor, device_id)
                    layer_weight_bytes += int(block_state[key].nbytes)
            torch.cuda.set_device(device_id)
            dsa_objects[device_id].exec_seq[layer_idx].init_tilert_weights(block_state)
            del block_state

        elapsed = time.perf_counter() - layer_start
        layer_reports.append(
            {
                "layer": layer_idx,
                "kind": "dense" if layer_idx < model_args.n_dense_layers else "moe",
                "seconds": elapsed,
                "weights_bytes": layer_weight_bytes,
            }
        )
        print(
            "[blaise-stream-step] "
            f"layer={layer_idx} kind={layer_reports[-1]['kind']} seconds={elapsed:.2f}",
            flush=True,
        )
        del ffn_weights
        del mla_weights
        del compat_tensors
        del layer_tensors
        torch.cuda.empty_cache()

    special_start = time.perf_counter()
    _stage_log("special-start")
    torch.cuda.set_device(conversion_device)
    _stage_log("special-load-head-norm-begin", conversion_device=conversion_device)
    norm_weight = _load_tensor(args.model_dir, weight_map, "model.norm.weight", conversion_device)
    head_weight = _load_tensor(args.model_dir, weight_map, "lm_head.weight", conversion_device)
    _sync_device(conversion_device, "special-load-head-norm")
    _stage_log("special-load-head-norm-end")

    _stage_log("special-shard-head-norm-begin")
    head_converter = RMSNormHeadProj(model_args, device_id=0, num_devices=num_devices)
    norm_shards, head_shards = head_converter.device_sharding(
        {
            "model.norm.weight": norm_weight,
            "lm_head.weight": head_weight,
        }
    )
    _stage_log("special-shard-head-norm-end", shard_count=len(head_shards))
    for device_id in range(num_devices):
        _stage_log("special-head-init-begin", device=device_id)
        head_state = {
            "model.norm.weight": _to_device_tensor(norm_shards[device_id], device_id),
            "lm_head.weight": _to_device_tensor(head_shards[device_id], device_id),
        }
        torch.cuda.set_device(device_id)
        dsa_objects[device_id].exec_seq[-1].init_tilert_weights(head_state)
        _sync_device(device_id, "special-head-init")
        _stage_log(
            "special-head-init-end",
            device=device_id,
            bytes=_tensor_list_bytes(list(head_state.values())),
        )
        del head_state

    _stage_log("special-load-embed-begin", conversion_device=conversion_device)
    embed_weight = _load_tensor(
        args.model_dir, weight_map, "model.embed_tokens.weight", conversion_device
    )
    _sync_device(conversion_device, "special-load-embed")
    _stage_log("special-load-embed-end", bytes=int(embed_weight.nbytes))
    for device_id, dsa in enumerate(dsa_objects):
        _stage_log("special-embed-copy-begin", device=device_id)
        dsa.embed_tokens_weight = _to_device_tensor(embed_weight, device_id)
        _sync_device(device_id, "special-embed-copy")
        _stage_log("special-freqs-begin", device=device_id)
        dsa.freqs_cis = _freqs_cis(model_args, device_id)
        _sync_device(device_id, "special-freqs")
        _stage_log("special-embed-freqs-end", device=device_id)

    mtp_objects = None
    mtp_seconds = 0.0
    if mtp_capacity_sweep:
        mtp_objects, mtp_seconds = _build_grafted_mtp_objects(
            args,
            model_args,
            converter,
            weight_map,
            dsa_objects,
            norm_shards,
            head_shards,
        )

    special_seconds = time.perf_counter() - special_start
    _stage_log("special-end", seconds=f"{special_seconds:.2f}")
    del embed_weight
    del head_weight
    del norm_weight
    del head_shards
    del norm_shards
    torch.cuda.empty_cache()

    init_seconds = time.perf_counter() - start
    sweep_reports = []
    expected_report_count = len(sweep_values) * len(_sampling_top_ks(args))
    for sweep_value in sweep_values:
        if mtp_capacity_sweep:
            batch_size = 1
            forward_seq_len = args.mtp_seq_len
            native_tokens_per_step = args.mtp_seq_len
        elif maxseq_sweep:
            _stage_log("runtime-max-seq-len-begin", max_seq_len=sweep_value)
            _set_runtime_max_seq_len(model_args, dsa_objects, sweep_value)
            _stage_log("runtime-max-seq-len-end", max_seq_len=sweep_value)
            batch_size = 1
            forward_seq_len = 1
            native_tokens_per_step = 1
        elif seqlen_sweep or packed_concurrency_sweep:
            batch_size = 1
            forward_seq_len = sweep_value
            native_tokens_per_step = forward_seq_len
        elif serial_context_sweep:
            batch_size = 1
            forward_seq_len = 1
            native_tokens_per_step = sweep_value
        else:
            batch_size = sweep_value
            forward_seq_len = args.forward_seq_len
            native_tokens_per_step = batch_size
        _stage_log(
            "sweep-point-begin",
            batch_size=batch_size,
            forward_seq_len=forward_seq_len,
        )
        device_results, prepare_seconds = _prepare_blaise_runtime(
            args,
            model_args,
            dsa_objects,
            batch_size,
            forward_seq_len,
            mtp_objects if mtp_capacity_sweep else None,
        )

        for sampling_idx, sampling_top_k in enumerate(_sampling_top_ks(args)):
            if sampling_idx > 0:
                if mtp_capacity_sweep:
                    dsa_show_hands_reset(True, False)
                    dsa_show_hands_reset(False, False)
                else:
                    dsa_show_hands_reset(False, False)
            _stage_log("sampling-point-begin", top_k=sampling_top_k)
            _set_sampling_config(args, device_results, sampling_top_k)
            dsa_show_hands_set_sampling_seed(args.sampling_seed, mtp_capacity_sweep, False)
            token = _make_decode_token(
                args,
                batch_size,
                forward_seq_len,
                packed_concurrency_sweep,
            )
            mtp_draft_tokens = None
            mtp_accepted_counts = []
            mtp_predicted_sample = []
            mtp_prefill_times = []
            mtp_prefill_valid_counts = []
            if mtp_capacity_sweep:
                _set_mtp_decode_state(args, device_results, args.mtp_start_pos)
                mtp_prefill_times, mtp_prefill_valid_counts = _run_mtp_prefill(
                    args,
                    device_results,
                )
                mtp_draft_tokens = _make_mtp_draft_tokens(args)
            request_cache_states = None
            request_cur_positions = None
            request_tokens = None
            if serial_context_sweep:
                if sweep_value > 1:
                    _stage_log("serial-context-cache-clone-begin", concurrency=sweep_value)
                    request_cache_states = _clone_cache_states(device_results, sweep_value)
                    _stage_log(
                        "serial-context-cache-clone-end",
                        concurrency=sweep_value,
                        cache_bytes_per_request=_cache_state_bytes(device_results),
                        copy_mode="prefix",
                    )
                else:
                    request_cache_states = None
                    _stage_log("serial-context-native-single", concurrency=sweep_value)
                request_cur_positions = [args.serial_context_start_pos] * sweep_value
                request_tokens = [
                    torch.tensor(args.token_id, dtype=torch.long) for _ in range(sweep_value)
                ]

            forward_times = []
            cuda_event_times: dict[str, list[float]] = {}
            token_out_sample = []
            profiler_started = False
            try:
                for step_idx in range(args.warmup_steps + args.measure_steps):
                    if step_idx == args.warmup_steps:
                        _stage_log(
                            "sweep-measure-begin",
                            batch_size=batch_size,
                            forward_seq_len=forward_seq_len,
                            top_k=sampling_top_k,
                        )
                        if args.dump_profile_logs:
                            _stage_log("profile-log-clear-begin")
                            _clear_profile_logs(args, device_results)
                            _stage_log("profile-log-clear-end")
                        if args.cuda_profiler_range:
                            _stage_log("cuda-profiler-start")
                            _cuda_profiler_start()
                            profiler_started = True
                    step_start = time.perf_counter()
                    start_events, end_events, event_device_ids = _record_forward_events(
                        args, num_devices
                    )
                    if serial_context_sweep:
                        assert request_cur_positions is not None
                        assert request_tokens is not None
                        token_out_values = _run_serial_context_round(
                            args,
                            device_results,
                            request_cache_states,
                            request_cur_positions,
                            request_tokens,
                        )
                        token_out_sample = token_out_values[:8]
                    elif mtp_capacity_sweep:
                        assert mtp_draft_tokens is not None
                        accepted, predicted, mtp_draft_tokens = _run_mtp_capacity_forward(
                            args,
                            device_results,
                            mtp_draft_tokens,
                        )
                        if step_idx >= args.warmup_steps:
                            mtp_accepted_counts.append(accepted)
                        mtp_predicted_sample = predicted
                    else:
                        dsa_show_hands(token, False, False)
                        _sync_forward(args, num_devices)
                    event_times = _finish_forward_events(
                        start_events,
                        end_events,
                        event_device_ids,
                    )
                    elapsed = time.perf_counter() - step_start
                    if step_idx >= args.warmup_steps:
                        forward_times.append(elapsed)
                        for device_id, event_time in event_times.items():
                            cuda_event_times.setdefault(device_id, []).append(event_time)
            finally:
                if profiler_started:
                    _cuda_profiler_stop()
                    _stage_log("cuda-profiler-stop")

            if args.forward_sync != "all":
                for device_id in range(num_devices):
                    torch.cuda.synchronize(device_id)
            profile_log_summary = _summarize_profile_logs(args, device_results)

            p50 = _quantile(forward_times, 0.50)
            p95 = _quantile(forward_times, 0.95)
            accepted_p50 = (
                _quantile([float(value) for value in mtp_accepted_counts], 0.50)
                if mtp_accepted_counts else 1.0
            )
            accepted_mean = (
                sum(mtp_accepted_counts) / len(mtp_accepted_counts)
                if mtp_accepted_counts else 1.0
            )
            effective_tok_s = (
                sum(mtp_accepted_counts) / sum(forward_times)
                if mtp_accepted_counts and sum(forward_times) > 0.0
                else 1.0 / p50
            )
            cuda_event_p50 = {
                device_id: _quantile(times, 0.50)
                for device_id, times in cuda_event_times.items()
            }
            cuda_event_p95 = {
                device_id: _quantile(times, 0.95)
                for device_id, times in cuda_event_times.items()
            }
            token_out_tensor = device_results[0][0][Idx.TOKEN_OUT]
            sample_count = min(native_tokens_per_step, 8)
            if mtp_capacity_sweep:
                token_out_sample = mtp_predicted_sample[:sample_count]
            elif serial_context_sweep:
                token_out_sample = token_out_sample[:sample_count]
            elif seqlen_sweep or packed_concurrency_sweep:
                token_out_sample = [
                    int(token_out_tensor[0, col, 0].item()) for col in range(sample_count)
                ]
            else:
                token_out_sample = [
                    int(token_out_tensor[row, 0, 0].item()) for row in range(sample_count)
                ]
            semantics = _measurement_semantics(
                serial_context_sweep,
                packed_concurrency_sweep,
                seqlen_sweep,
                maxseq_sweep,
                mtp_capacity_sweep,
                sweep_value,
            )
            report = {
                "concurrency": native_tokens_per_step,
                "batch_size": batch_size,
                "forward_seq_len": forward_seq_len,
                "runtime_max_seq_len": model_args.max_seq_len,
                **semantics,
                "sampling_top_k": sampling_top_k,
                "sampling_top_p": args.top_p,
                "sampling_temperature": args.temperature,
                "sampling_use_topp": args.use_topp,
                "packed_concurrency": packed_concurrency_sweep,
                "serial_context_swap": serial_context_sweep,
                "serial_context_copy_mode": (
                    "native-single" if serial_context_sweep and sweep_value == 1
                    else "prefix" if serial_context_sweep else "none"
                ),
                "native_tokens_per_step": native_tokens_per_step,
                "native_forward_calls_per_round": (
                    native_tokens_per_step if serial_context_sweep else 1
                ),
                "mtp_graft_layer": args.mtp_graft_layer if mtp_capacity_sweep else None,
                "mtp_seq_len": args.mtp_seq_len if mtp_capacity_sweep else None,
                "mtp_prefill_len": args.mtp_prefill_len if mtp_capacity_sweep else None,
                "mtp_prefill_times_s": mtp_prefill_times,
                "mtp_prefill_valid_token_counts": mtp_prefill_valid_counts,
                "mtp_feed_next_draft": args.mtp_feed_next_draft if mtp_capacity_sweep else None,
                "mtp_accepted_counts": mtp_accepted_counts,
                "mtp_accepted_p50": accepted_p50 if mtp_capacity_sweep else None,
                "mtp_accepted_mean": accepted_mean if mtp_capacity_sweep else None,
                "prepare_seconds": prepare_seconds,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "forward_sync": args.forward_sync,
                "forward_times_s": forward_times,
                "forward_p50_s": p50,
                "forward_p95_s": p95,
                "cuda_event_devices": (
                    args.cuda_event_devices if args.measure_cuda_events else "none"
                ),
                "cuda_profiler_range": args.cuda_profiler_range,
                "cuda_event_p50_s": cuda_event_p50,
                "cuda_event_p95_s": cuda_event_p95,
                "profile_logs": profile_log_summary,
                "tok_s_user_p50": (
                    accepted_p50 / p50 if mtp_capacity_sweep else 1.0 / p50
                ),
                "agg_out_tok_s_p50": (
                    accepted_p50 / p50
                    if mtp_capacity_sweep else native_tokens_per_step / p50
                ),
                "effective_tok_s_user": effective_tok_s,
                "token_out_device0_sample": token_out_sample,
                "cache_bytes_per_logical_request": (
                    _cache_state_bytes(device_results)
                    if serial_context_sweep and sweep_value > 1 else 0
                ),
                "memory": _memory_reports(num_devices),
            }
            sweep_reports.append(report)
            _stage_log(
                "sampling-point-end",
                batch_size=batch_size,
                forward_seq_len=forward_seq_len,
                top_k=sampling_top_k,
                p50_ms=f"{p50 * 1000.0:.3f}",
                tok_s_user=f"{report['tok_s_user_p50']:.2f}",
                agg_out_tok_s=f"{report['agg_out_tok_s_p50']:.2f}",
            )

        if mtp_capacity_sweep:
            dsa_show_hands_reset(True, False)
            dsa_show_hands_reset(False, False)
            dsa_show_hands_go_home(True, False)
            dsa_show_hands_go_home(False, False)
        else:
            dsa_show_hands_reset(False, False)
            dsa_show_hands_go_home(False, False)
        del request_cache_states
        del request_cur_positions
        del request_tokens
        del device_results
        torch.cuda.empty_cache()
        if args.write_json is not None:
            partial_result = {
                "mode": args.mode,
                "tilert": tilert.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "visible_cuda_devices": visible_devices,
                "num_devices_arg": num_devices,
                "conversion_device": conversion_device,
                "model_args": _model_args_summary(model_args),
                "init_seconds": init_seconds,
                "special_seconds": special_seconds,
                "mtp_seconds": mtp_seconds,
                "sweep_concurrencies": _parse_int_list(args.sweep_concurrencies),
                "sweep_seq_lens": _parse_int_list(args.sweep_seq_lens),
                "sweep_max_seq_lens": _runtime_max_seq_lens(args, model_args),
                "sweep_top_ks": _sampling_top_ks(args),
                "layer_reports": layer_reports,
                "sweep": sweep_reports,
                "partial": len(sweep_reports) != expected_report_count,
            }
            args.write_json.parent.mkdir(parents=True, exist_ok=True)
            args.write_json.write_text(json.dumps(partial_result, indent=2), encoding="utf-8")

    return {
        "mode": args.mode,
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": visible_devices,
        "num_devices_arg": num_devices,
        "conversion_device": conversion_device,
        "model_args": _model_args_summary(model_args),
        "init_seconds": init_seconds,
        "special_seconds": special_seconds,
        "mtp_seconds": mtp_seconds,
        "sweep_concurrencies": _parse_int_list(args.sweep_concurrencies),
        "sweep_seq_lens": _parse_int_list(args.sweep_seq_lens),
        "sweep_max_seq_lens": _runtime_max_seq_lens(args, model_args),
        "sweep_top_ks": _sampling_top_ks(args),
        "layer_reports": layer_reports,
        "sweep": sweep_reports,
    }


def probe_blaise_stream_step(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import tilert
    from tilert.models.deepseek_v3_2.modules.dsa import Dsa
    from tilert.models.deepseek_v3_2.modules.end2end import (
        dsa_show_hands,
        dsa_show_hands_prepare_money,
        dsa_show_hands_set_sampling_seed,
    )
    from tilert.models.deepseek_v3_2.ops.rmsnorm_head_proj import RMSNormHeadProj
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx
    from tilert.models.preprocess.weight_converter import WeightConverter
    from tilert.tilert_init import tilert_init
    from tilert.utils import get_profile_log_tensor

    _load_backend()
    _patch_flexible_tilert_weight_dequant()
    visible_devices = torch.cuda.device_count()
    num_devices = args.num_devices or visible_devices
    if visible_devices < num_devices:
        raise ValueError(f"Need {num_devices} visible CUDA devices, got {visible_devices}")
    if args.block_size != 128:
        raise ValueError("blaise-stream-step requires --block-size 128 for TileRT FP8 compat")

    model_args = _build_model_args(args)
    conversion_device = args.conversion_device
    if conversion_device < 0 or conversion_device >= visible_devices:
        raise ValueError(f"Invalid conversion device {conversion_device}")

    _stage_log("tilert-init-begin")
    tilert_init()
    for device_id in range(num_devices):
        _sync_device(device_id, "tilert-init")
    _stage_log("tilert-init-end")

    weight_map = _weight_map(args.model_dir)
    converter = WeightConverter(
        model_args=model_args,
        num_devices=num_devices,
        model_dir=str(args.model_dir),
        save_dir="/tmp/blaise-tilert-unused",
        test_mode=True,
    )

    start = time.perf_counter()
    dsa_objects = []
    for device_id in range(num_devices):
        torch.cuda.set_device(device_id)
        dsa_objects.append(Dsa(model_args, device_id=device_id, num_devices=num_devices))

    layer_reports = []
    for layer_idx in range(model_args.n_layers):
        layer_start = time.perf_counter()
        torch.cuda.set_device(conversion_device)
        layer_tensors = _load_layer_tensors(
            args.model_dir, weight_map, layer_idx, conversion_device
        )
        compat_tensors = _add_tilert_fp8_aliases(layer_tensors)
        mla_weights = converter.transform_mla(compat_tensors, layer_idx)
        if layer_idx < model_args.n_dense_layers:
            ffn_weights = converter.transform_mlp(compat_tensors, layer_idx)
        else:
            ffn_weights = converter.transform_moe(compat_tensors, layer_idx)
            ffn_weights = _pad_moe_tilert_weights_for_native_abi(
                ffn_weights,
                args.native_expert_weight_pad,
            )

        layer_weight_bytes = 0
        for device_id in range(num_devices):
            dev_key = f"dev_{device_id}"
            block_state = {}
            for source in (mla_weights[dev_key], ffn_weights[dev_key]):
                for key, tensor in source.items():
                    block_state[key] = _to_device_tensor(tensor, device_id)
                    layer_weight_bytes += int(block_state[key].nbytes)
            torch.cuda.set_device(device_id)
            dsa_objects[device_id].exec_seq[layer_idx].init_tilert_weights(block_state)
            del block_state

        elapsed = time.perf_counter() - layer_start
        layer_reports.append(
            {
                "layer": layer_idx,
                "kind": "dense" if layer_idx < model_args.n_dense_layers else "moe",
                "seconds": elapsed,
                "weights_bytes": layer_weight_bytes,
            }
        )
        print(
            "[blaise-stream-step] "
            f"layer={layer_idx} kind={layer_reports[-1]['kind']} seconds={elapsed:.2f}",
            flush=True,
        )
        del ffn_weights
        del mla_weights
        del compat_tensors
        del layer_tensors
        torch.cuda.empty_cache()

    special_start = time.perf_counter()
    _stage_log("special-start")
    torch.cuda.set_device(conversion_device)
    _stage_log("special-load-head-norm-begin", conversion_device=conversion_device)
    norm_weight = _load_tensor(args.model_dir, weight_map, "model.norm.weight", conversion_device)
    head_weight = _load_tensor(args.model_dir, weight_map, "lm_head.weight", conversion_device)
    _sync_device(conversion_device, "special-load-head-norm")
    _stage_log("special-load-head-norm-end")

    _stage_log("special-shard-head-norm-begin")
    head_converter = RMSNormHeadProj(model_args, device_id=0, num_devices=num_devices)
    norm_shards, head_shards = head_converter.device_sharding(
        {
            "model.norm.weight": norm_weight,
            "lm_head.weight": head_weight,
        }
    )
    _stage_log("special-shard-head-norm-end", shard_count=len(head_shards))
    for device_id in range(num_devices):
        _stage_log("special-head-init-begin", device=device_id)
        head_state = {
            "model.norm.weight": _to_device_tensor(norm_shards[device_id], device_id),
            "lm_head.weight": _to_device_tensor(head_shards[device_id], device_id),
        }
        torch.cuda.set_device(device_id)
        dsa_objects[device_id].exec_seq[-1].init_tilert_weights(head_state)
        _sync_device(device_id, "special-head-init")
        _stage_log(
            "special-head-init-end",
            device=device_id,
            bytes=_tensor_list_bytes(list(head_state.values())),
        )
        del head_state

    _stage_log("special-load-embed-begin", conversion_device=conversion_device)
    embed_weight = _load_tensor(
        args.model_dir, weight_map, "model.embed_tokens.weight", conversion_device
    )
    _sync_device(conversion_device, "special-load-embed")
    _stage_log("special-load-embed-end", bytes=int(embed_weight.nbytes))
    for device_id, dsa in enumerate(dsa_objects):
        _stage_log("special-embed-copy-begin", device=device_id)
        dsa.embed_tokens_weight = _to_device_tensor(embed_weight, device_id)
        _sync_device(device_id, "special-embed-copy")
        _stage_log("special-freqs-begin", device=device_id)
        dsa.freqs_cis = _freqs_cis(model_args, device_id)
        _sync_device(device_id, "special-freqs")
        _stage_log("special-embed-freqs-end", device=device_id)
    special_seconds = time.perf_counter() - special_start
    _stage_log("special-end", seconds=f"{special_seconds:.2f}")
    del embed_weight
    del head_weight
    del norm_weight
    del head_shards
    del norm_shards
    torch.cuda.empty_cache()

    device_results = []
    p2p: dict[int, dict[str, Any]] = {}
    prep_start = time.perf_counter()
    _stage_log("prep-start")
    for device_id, dsa in enumerate(dsa_objects):
        torch.cuda.set_device(device_id)
        _stage_log("prep-temp-vars-begin", device=device_id)
        temp_vars = dsa.get_temp_vars(
            batch_size=args.batch_size,
            seq_len=args.forward_seq_len,
            extra_args={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_topp": args.use_topp,
            },
        )
        _sync_device(device_id, "prep-temp-vars")
        _stage_log(
            "prep-temp-vars-end",
            device=device_id,
            count=len(temp_vars),
            token_out_shape=_shape(temp_vars[Idx.TOKEN_OUT]),
        )
        _pad_runtime_abi_buffers(args, model_args, device_id, dsa, temp_vars)

        _stage_log("prep-continuous-storage-begin", device=device_id)
        intermediates = _continuous_storage(temp_vars, device_id)
        _sync_device(device_id, "prep-continuous-storage")
        _stage_log(
            "prep-continuous-storage-end",
            device=device_id,
            count=len(intermediates),
            bytes=_tensor_list_bytes(intermediates),
        )

        _stage_log("prep-sampling-config-copy-begin", device=device_id)
        intermediates[Idx.SAMPLING_CONFIG].copy_(
            torch.tensor(
                [args.temperature, args.top_p, float(args.top_k), 1.0 if args.use_topp else 0.0],
                dtype=torch.float32,
                device=device_id,
            )
        )
        _sync_device(device_id, "prep-sampling-config-copy")

        _stage_log("prep-cache-vars-begin", device=device_id)
        caches = dsa.get_cache_vars()
        _sync_device(device_id, "prep-cache-vars")
        _stage_log(
            "prep-cache-vars-end",
            device=device_id,
            count=len(caches),
            first_shape=_shape(caches[0]) if caches else [],
            bytes=_tensor_list_bytes(caches),
        )

        _stage_log("prep-params-begin", device=device_id)
        params = dsa.get_weights_list()
        _stage_log(
            "prep-params-end",
            device=device_id,
            count=len(params),
            bytes=_tensor_list_bytes(params),
        )

        _stage_log("prep-profile-log-begin", device=device_id)
        profile_logs = get_profile_log_tensor(
            device=device_id,
            num_max_insts=args.profile_log_num_max_insts,
        )
        _sync_device(device_id, "prep-profile-log")
        _stage_log("prep-profile-log-end", device=device_id, shape=_shape(profile_logs))

        if device_id == 0:
            p2p[device_id] = {"peer_bufs": dsa.v2_peer_bufs}
            _stage_log(
                "prep-p2p-recorded",
                device=device_id,
                peer_bufs_ptr=int(dsa.v2_peer_bufs.data_ptr()),
            )
        else:
            p2p[device_id] = {"ll_buf": dsa.v2_ll_buf}
            _stage_log(
                "prep-p2p-recorded",
                device=device_id,
                ll_buf_ptr=int(dsa.v2_ll_buf.data_ptr()),
            )
        device_results.append((intermediates, caches, params, profile_logs))
        _stage_log("prep-device-end", device=device_id)

    _stage_log("p2p-peer-buffer-build-begin")
    peer_bufs_cpu = torch.zeros(num_devices - 1, dtype=torch.int64)
    for device_idx in range(num_devices - 1):
        peer_device = device_idx + 1
        peer_bufs_cpu[device_idx] = p2p[peer_device]["ll_buf"].data_ptr()
        _stage_log(
            "p2p-peer-buffer-entry",
            peer_device=peer_device,
            ptr=int(peer_bufs_cpu[device_idx].item()),
        )
    _stage_log("p2p-peer-buffer-copy-begin")
    p2p[0]["peer_bufs"].copy_(peer_bufs_cpu)
    _sync_device(0, "p2p-peer-buffer-copy")
    _stage_log("p2p-peer-buffer-copy-end")

    _stage_log("prepare-money-all-begin")
    for device_id, result in enumerate(device_results):
        torch.cuda.set_device(device_id)
        intermediates, caches, params, profile_logs = result
        _stage_log(
            "prepare-money-begin",
            device=device_id,
            params=len(params),
            intermediates=len(intermediates),
            caches=len(caches),
        )
        dsa_show_hands_prepare_money(
            params,
            intermediates,
            caches,
            profile_logs,
            args.forward_seq_len,
            False,
            False,
        )
        _sync_device(device_id, "prepare-money")
        _stage_log("prepare-money-end", device=device_id)
    _stage_log("prepare-money-cross-sync-begin")
    for device_id in range(num_devices):
        _sync_device(device_id, "prepare-money-cross-sync")
    _stage_log("prepare-money-all-end")
    prepare_seconds = time.perf_counter() - prep_start

    forward_start = time.perf_counter()
    _stage_log("sampling-seed-begin", seed=args.sampling_seed)
    dsa_show_hands_set_sampling_seed(args.sampling_seed, False, False)
    _stage_log("sampling-seed-end", seed=args.sampling_seed)
    _stage_log("forward-token-create-begin", token_id=args.token_id)
    token = torch.tensor(args.token_id, dtype=torch.long)
    _stage_log("forward-begin", token_device=token.device)
    dsa_show_hands(token, False, False)
    _stage_log("forward-launched")
    for device_id in range(num_devices):
        _sync_device(device_id, "forward")
    _stage_log("forward-end")
    forward_seconds = time.perf_counter() - forward_start

    token_out = int(device_results[0][0][Idx.TOKEN_OUT][0][0].item())
    memory_reports = []
    for device_id in range(num_devices):
        memory_reports.append(
            {
                "device": device_id,
                "allocated_bytes": torch.cuda.memory_allocated(device_id),
                "reserved_bytes": torch.cuda.memory_reserved(device_id),
            }
        )
    return {
        "mode": "blaise-stream-step",
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": visible_devices,
        "num_devices_arg": num_devices,
        "conversion_device": conversion_device,
        "model_args": _model_args_summary(model_args),
        "init_seconds": time.perf_counter() - start,
        "special_seconds": special_seconds,
        "prepare_seconds": prepare_seconds,
        "forward_seconds": forward_seconds,
        "token_out_device0": token_out,
        "layer_reports": layer_reports,
        "memory": memory_reports,
    }


def _continuous_storage(temp_vars: list[Any], device: int, aligned_size: int = 1024) -> list[Any]:
    import torch

    total_size = 0
    for temp_var in temp_vars:
        aligned_size_bytes = (temp_var.nbytes + aligned_size - 1) // aligned_size * aligned_size
        total_size += aligned_size_bytes
    large_tensor = torch.zeros(total_size, device=device, dtype=torch.uint8)
    cloned_vars = []
    offset = 0
    for temp_var in temp_vars:
        aligned_size_bytes = (temp_var.nbytes + aligned_size - 1) // aligned_size * aligned_size
        temp_storage = large_tensor[offset : offset + temp_var.nbytes]
        cloned_vars.append(temp_storage.view(temp_var.dtype).view(temp_var.shape))
        offset += aligned_size_bytes
    return cloned_vars


def _freqs_cis(model_args: Any, device: int) -> Any:
    import torch
    from tilert.models.utils import precompute_freqs_cis

    freqs_cis = precompute_freqs_cis(model_args)
    return torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], -1).to(device)


def probe_backend(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import tilert

    _load_backend()
    model_args = _build_model_args(args)
    return {
        "mode": "backend-load",
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "model_args": _model_args_summary(model_args),
    }


def probe_dsa_construct(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import tilert
    from tilert.models.deepseek_v3_2.modules.dsa import Dsa
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    _load_backend()
    visible_devices = torch.cuda.device_count()
    devices = _parse_devices(args.devices, visible_devices)
    model_args = _build_model_args(args)
    num_devices = args.num_devices or visible_devices
    device_reports = []
    for device_id in devices:
        torch.cuda.set_device(device_id)
        before_allocated = torch.cuda.memory_allocated(device_id)
        before_reserved = torch.cuda.memory_reserved(device_id)
        dsa = Dsa(model_args, device_id=device_id, num_devices=num_devices)
        temp_vars = dsa.get_temp_vars(
            batch_size=args.batch_size,
            seq_len=args.forward_seq_len,
            extra_args={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_topp": args.use_topp,
            },
        )
        cache_vars = dsa.get_cache_vars()
        after_allocated = torch.cuda.memory_allocated(device_id)
        after_reserved = torch.cuda.memory_reserved(device_id)
        device_reports.append(
            {
                "device": device_id,
                "name": torch.cuda.get_device_name(device_id),
                "allocated_delta_bytes": after_allocated - before_allocated,
                "reserved_delta_bytes": after_reserved - before_reserved,
                "temp_var_count": len(temp_vars),
                "cache_var_count": len(cache_vars),
                "scores_shape": _shape(temp_vars[Idx.SCORES]),
                "idx_selects_shape": _shape(temp_vars[Idx.IDX_SELECTS]),
                "up_gate_shape": _shape(temp_vars[Idx.UP_GATE]),
                "first_cache_shapes": [_shape(value) for value in cache_vars[:3]],
            }
        )
        del cache_vars
        del temp_vars
        del dsa
        torch.cuda.empty_cache()
    return {
        "mode": "dsa-construct",
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": visible_devices,
        "num_devices_arg": num_devices,
        "model_args": _model_args_summary(model_args),
        "devices": device_reports,
    }


def probe_state_dict_contract(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import tilert
    from tilert.models.deepseek_v3_2.modules.dsa import Dsa

    _load_backend()
    model_args = _build_model_args(args)
    visible_devices = torch.cuda.device_count()
    num_devices = args.num_devices or visible_devices
    if visible_devices < num_devices:
        raise ValueError(f"Need {num_devices} visible CUDA devices, got {visible_devices}")

    per_device = []
    full_contract = {}
    for device_id in range(num_devices):
        torch.cuda.set_device(device_id)
        dsa = Dsa(model_args, device_id=device_id, num_devices=num_devices)
        layer_keys = []
        blocks = []
        for block, prefix, suffix in zip(dsa.exec_seq, dsa.prefix_seq, dsa.suffix_seq):
            keys = [f"{prefix}{alias}{suffix}" for alias in block.get_tilert_weights_alias()]
            layer_keys.extend(keys)
            blocks.append(
                {
                    "block_type": type(block).__name__,
                    "prefix": prefix,
                    "suffix": suffix,
                    "key_count": len(keys),
                    "keys": keys,
                }
            )
        extra_keys = [
            "model.embed_tokens.weight",
            f"layer_{model_args.n_layers}_lm_head.weight_dev_{device_id}",
            f"layer_{model_args.n_layers}_model.norm.weight_dev_{device_id}",
        ]
        all_keys = [*layer_keys, *extra_keys]
        full_contract[f"dev_{device_id}"] = {
            "keys": all_keys,
            "blocks": blocks,
            "extra_keys": extra_keys,
        }
        per_device.append(
            {
                "device": device_id,
                "name": torch.cuda.get_device_name(device_id),
                "layer_key_count": len(layer_keys),
                "extra_key_count": len(extra_keys),
                "total_key_count": len(all_keys),
                "first_keys": all_keys[:20],
                "last_keys": all_keys[-10:],
            }
        )
        del dsa
        torch.cuda.empty_cache()

    result = {
        "mode": "state-dict-contract",
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": visible_devices,
        "num_devices_arg": num_devices,
        "model_args": _model_args_summary(model_args),
        "total_key_count": sum(item["total_key_count"] for item in per_device),
        "per_device": per_device,
    }
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(
            json.dumps({**result, "contract": full_contract}, indent=2),
            encoding="utf-8",
        )
        result["write_json"] = str(args.write_json)
    return result


def probe_dsa_random_step(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_random_weight_init:
        raise ValueError("dsa-random-step requires --confirm-random-weight-init")

    import torch
    import tilert
    from tilert.models.deepseek_v3_2.modules.dsa import Dsa
    from tilert.models.deepseek_v3_2.modules.end2end import (
        dsa_show_hands,
        dsa_show_hands_prepare_money,
    )
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx
    from tilert.utils import get_profile_log_tensor

    _load_backend()
    if args.patch_flexible_weight_dequant:
        _patch_flexible_tilert_weight_dequant()
    model_args = _build_model_args(args)
    visible_devices = torch.cuda.device_count()
    num_devices = args.num_devices or visible_devices
    if visible_devices < num_devices:
        raise ValueError(f"Need {num_devices} visible CUDA devices, got {visible_devices}")

    start = time.perf_counter()
    device_results = []
    dsa_objects = []
    p2p: dict[int, dict[str, Any]] = {}
    for device_id in range(num_devices):
        torch.cuda.set_device(device_id)
        dsa = Dsa(model_args, device_id=device_id, num_devices=num_devices)
        dsa.init_random_weights()
        dsa.embed_tokens_weight = torch.empty(
            model_args.vocab_size,
            model_args.dim,
            dtype=torch.bfloat16,
            device=device_id,
        )
        dsa.freqs_cis = _freqs_cis(model_args, device_id)
        temp_vars = dsa.get_temp_vars(
            batch_size=args.batch_size,
            seq_len=args.forward_seq_len,
            extra_args={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_topp": args.use_topp,
            },
        )
        intermediates = _continuous_storage(temp_vars, device_id)
        intermediates[Idx.SAMPLING_CONFIG].copy_(
            torch.tensor(
                [args.temperature, args.top_p, float(args.top_k), 1.0 if args.use_topp else 0.0],
                dtype=torch.float32,
                device=device_id,
            )
        )
        caches = dsa.get_cache_vars()
        params = dsa.get_weights_list()
        profile_logs = get_profile_log_tensor(
            device=device_id,
            num_max_insts=args.profile_log_num_max_insts,
        )
        if device_id == 0:
            p2p[device_id] = {"peer_bufs": dsa.v2_peer_bufs}
        else:
            p2p[device_id] = {"ll_buf": dsa.v2_ll_buf}
        device_results.append((intermediates, caches, params, profile_logs))
        dsa_objects.append(dsa)
    for device_id in range(num_devices):
        torch.cuda.synchronize(device_id)
    init_seconds = time.perf_counter() - start

    peer_bufs_cpu = torch.zeros(num_devices - 1, dtype=torch.int64)
    for device_idx in range(num_devices - 1):
        peer_device = device_idx + 1
        peer_bufs_cpu[device_idx] = p2p[peer_device]["ll_buf"].data_ptr()
    p2p[0]["peer_bufs"].copy_(peer_bufs_cpu)

    prepare_start = time.perf_counter()
    for device_id, result in enumerate(device_results):
        torch.cuda.set_device(device_id)
        intermediates, caches, params, profile_logs = result
        dsa_show_hands_prepare_money(
            params,
            intermediates,
            caches,
            profile_logs,
            args.forward_seq_len,
            False,
            False,
        )
    for device_id in range(num_devices):
        torch.cuda.synchronize(device_id)
    prepare_seconds = time.perf_counter() - prepare_start

    forward_start = time.perf_counter()
    dsa_show_hands(torch.tensor(args.token_id, dtype=torch.long), False, False)
    for device_id in range(num_devices):
        torch.cuda.synchronize(device_id)
    forward_seconds = time.perf_counter() - forward_start

    token_out = int(device_results[0][0][Idx.TOKEN_OUT][0][0].item())
    memory_reports = []
    for device_id in range(num_devices):
        memory_reports.append(
            {
                "device": device_id,
                "allocated_bytes": torch.cuda.memory_allocated(device_id),
                "reserved_bytes": torch.cuda.memory_reserved(device_id),
            }
        )
    return {
        "mode": "dsa-random-step",
        "tilert": tilert.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": visible_devices,
        "num_devices_arg": num_devices,
        "model_args": _model_args_summary(model_args),
        "init_seconds": init_seconds,
        "prepare_seconds": prepare_seconds,
        "forward_seconds": forward_seconds,
        "token_out_device0": token_out,
        "memory": memory_reports,
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
        "--mode",
        choices=(
            "backend-load",
            "dsa-construct",
            "state-dict-contract",
            "dsa-random-step",
            "blaise-layer-init",
            "blaise-stream-step",
            "blaise-stream-sweep",
            "blaise-stream-seqlen-sweep",
            "blaise-stream-packed-concurrency-sweep",
            "blaise-stream-serial-context-sweep",
            "blaise-stream-maxseq-sweep",
            "blaise-stream-mtp-capacity",
        ),
        default="backend-load",
        help="Runtime probe to run.",
    )
    parser.add_argument("--devices", help="Comma-separated visible CUDA device ids to probe.")
    parser.add_argument(
        "--num-devices",
        type=int,
        help="TileRT logical device count. Defaults to torch.cuda.device_count().",
    )
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--forward-seq-len",
        type=int,
        default=4,
        help="TileRT forward_max_seq_len used for scratch tensors.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Override ModelArgs.max_seq_len for smaller scratch/cache probes.",
    )
    parser.add_argument(
        "--sweep-max-seq-lens",
        default="1024,2048,4096",
        help="Comma-separated runtime max_seq_len values for blaise-stream-maxseq-sweep.",
    )
    parser.add_argument("--kv-cache-pad", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--use-topp", action="store_true")
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--sweep-top-ks",
        help=(
            "Optional comma-separated sampling top-k values to measure after one "
            "prepared TileRT runtime."
        ),
    )
    parser.add_argument(
        "--sweep-concurrencies",
        default="1,4,8,16,32,64",
        help="Comma-separated native batch sizes for blaise-stream-sweep.",
    )
    parser.add_argument(
        "--sweep-seq-lens",
        default="1,2,4",
        help="Comma-separated forward sequence lengths for blaise-stream-seqlen-sweep.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument(
        "--forward-sync",
        choices=("all", "device0", "none"),
        default="all",
        help="CUDA devices to synchronize after each measured native forward.",
    )
    parser.add_argument(
        "--measure-cuda-events",
        action="store_true",
        help="Record CUDA event elapsed times around each native forward.",
    )
    parser.add_argument(
        "--cuda-event-devices",
        choices=("all", "device0"),
        default="all",
        help="Devices on which to record CUDA forward timing events.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around measured forward iterations.",
    )
    parser.add_argument(
        "--serial-context-start-pos",
        type=int,
        default=0,
        help="Initial native cur_pos for each logical request in serial-context mode.",
    )
    parser.add_argument(
        "--serial-context-feed-output",
        action="store_true",
        help="Feed each request's previous sampled token into its next serial-context step.",
    )
    parser.add_argument(
        "--serial-context-set-cur-pos",
        action="store_true",
        help="Force dsa_show_hands_set_cur_pos before every serial-context native call.",
    )
    parser.add_argument(
        "--mtp-graft-layer",
        type=int,
        default=60,
        help="Existing MoE layer whose converted weights are grafted into the dummy MTP slot.",
    )
    parser.add_argument(
        "--mtp-seq-len",
        type=int,
        default=4,
        help="Draft-token lane count for the public TileRT MTP e2e path.",
    )
    parser.add_argument(
        "--mtp-start-pos",
        type=int,
        default=0,
        help="Initial CUR_POS written into MTP temp vars for capacity probes.",
    )
    parser.add_argument(
        "--mtp-prefill-len",
        type=int,
        default=0,
        help=(
            "Optional repeated-token MTP prefill length before measured decode; "
            "0 disables the prefill state-machine warmup."
        ),
    )
    parser.add_argument(
        "--mtp-feed-next-draft",
        action="store_true",
        help="Feed NEXT_DRAFT_TOKENS back into the next measured MTP call.",
    )
    parser.add_argument(
        "--profile-log-num-max-insts",
        type=int,
        default=65536,
        help="Rows to allocate in each TileRT profile-log tensor.",
    )
    parser.add_argument(
        "--dump-profile-logs",
        action="store_true",
        help="Summarize nonzero TileRT profile-log rows after measured forwards.",
    )
    parser.add_argument(
        "--profile-log-devices",
        choices=("device0", "all"),
        default="device0",
        help="Devices whose TileRT profile logs should be summarized.",
    )
    parser.add_argument(
        "--profile-log-max-rows",
        type=int,
        default=24,
        help="Maximum profile-log rows to copy for each row-selection group.",
    )
    parser.add_argument(
        "--profile-log-values-per-row",
        type=int,
        default=16,
        help="Maximum nonzero uint64 values to include for each copied profile-log row.",
    )
    parser.add_argument(
        "--native-index-topk-pad",
        type=int,
        default=0,
        help="Probe-only over-allocation for IDX_SELECTS and P2P ll_buf.",
    )
    parser.add_argument(
        "--native-routed-experts-pad",
        type=int,
        default=0,
        help="Probe-only over-allocation for routed-expert score buffers.",
    )
    parser.add_argument(
        "--native-expert-weight-pad",
        type=int,
        default=0,
        help="Probe-only padding for converted MoE tensors to public routed experts.",
    )
    parser.add_argument(
        "--native-idx-sel-ws-pad",
        type=int,
        help=(
            "Probe-only over-allocation for IDX_SEL_WS workspace. Do not use "
            "for public dsv32: native prepare expects exactly 200*1024 + 260."
        ),
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--target-device", type=int, default=0)
    parser.add_argument("--conversion-device", type=int, default=0)
    parser.add_argument(
        "--patch-flexible-weight-dequant",
        action="store_true",
        help="Patch TileRT's reference FP8 dequant helper for partial 128-row groups.",
    )
    parser.add_argument(
        "--confirm-random-weight-init",
        action="store_true",
        help="Required for dsa-random-step because it allocates random model weights.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Optional path for full JSON output in modes that support it.",
    )
    parser.add_argument(
        "--hard-exit-after-json",
        action="store_true",
        help="Use os._exit(0) after printing JSON to bypass TileRT teardown crashes.",
    )
    args = parser.parse_args()

    if args.mode == "backend-load":
        result = probe_backend(args)
    elif args.mode == "dsa-construct":
        result = probe_dsa_construct(args)
    elif args.mode == "state-dict-contract":
        result = probe_state_dict_contract(args)
    elif args.mode == "dsa-random-step":
        result = probe_dsa_random_step(args)
    elif args.mode == "blaise-layer-init":
        result = probe_blaise_layer_init(args)
    elif args.mode in (
        "blaise-stream-sweep",
        "blaise-stream-seqlen-sweep",
        "blaise-stream-packed-concurrency-sweep",
        "blaise-stream-serial-context-sweep",
        "blaise-stream-maxseq-sweep",
        "blaise-stream-mtp-capacity",
    ):
        result = probe_blaise_stream_sweep(args)
    else:
        result = probe_blaise_stream_step(args)
    print(json.dumps(result, indent=2), flush=True)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.hard_exit_after_json:
        os._exit(0)


if __name__ == "__main__":
    main()

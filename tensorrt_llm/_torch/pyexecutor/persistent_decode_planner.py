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

"""Eligibility planner for the experimental persistent decode path.

This module is intentionally side-effect free. It records the production
contracts a TileRT-style decode engine would have to consume directly:
scheduled requests, attention metadata, and KV ownership. The actual engine
must live below the scheduler, not behind the public TileRT batch-one ABI.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from tensorrt_llm.logger import logger

_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG"
_TIMING_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG"
_WINDOW_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG"


@dataclass(frozen=True)
class PersistentDecodePlan:
    eligible: bool
    reason: str
    model_type: str
    architectures: tuple[str, ...]
    real_context_requests: int
    real_generation_requests: int
    padded_context_requests: int
    padded_generation_requests: int
    metadata_contexts: int
    metadata_generations: int
    metadata_tokens: int
    metadata_seqs: int
    request_ids: tuple[int, ...]
    seq_lens: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    input_tokens: int
    cuda_graph_replay: bool
    cuda_graph_padding: bool
    enable_spec_decode: bool
    is_draft_model: bool
    kv_cache_manager_type: str
    attention_metadata_type: str


def persistent_decode_plan_debug_enabled() -> bool:
    return os.environ.get(_ENV_NAME, "0") == "1"


def persistent_decode_plan_enabled() -> bool:
    return (
        persistent_decode_plan_debug_enabled()
        or os.environ.get(_TIMING_ENV_NAME, "0") == "1"
        or os.environ.get(_WINDOW_ENV_NAME, "0") == "1")


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _tuple_from_maybe_tensor(value: object, limit: int | None = None) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
    except RuntimeError:
        return ()
    if isinstance(value, int):
        values = [value]
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError:
            return ()
    if limit is not None:
        values = values[:limit]
    return tuple(_as_int(item) for item in values)


def _request_ids_from_metadata(attn_metadata: Any, limit: int) -> tuple[int, ...]:
    request_ids = getattr(attn_metadata, "request_ids", None)
    return _tuple_from_maybe_tensor(request_ids, limit)


def _seq_lens_from_metadata(attn_metadata: Any, limit: int) -> tuple[int, ...]:
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    return _tuple_from_maybe_tensor(seq_lens, limit)


def _cached_tokens_from_metadata(attn_metadata: Any, limit: int) -> tuple[int, ...]:
    kv_cache_params = getattr(attn_metadata, "kv_cache_params", None)
    cached = getattr(kv_cache_params, "num_cached_tokens_per_seq", None)
    return _tuple_from_maybe_tensor(cached, limit)


def _input_token_count(inputs: dict[str, Any]) -> int:
    input_ids = inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is None:
        return 0
    try:
        if len(shape) == 0:
            return 1
        return int(shape[0])
    except (TypeError, ValueError):
        return 0


def _pretrained_config(model: Any) -> Any:
    model_config = getattr(model, "model_config", None)
    return getattr(model_config, "pretrained_config", None)


def _model_type(model: Any) -> str:
    config = _pretrained_config(model)
    return str(getattr(config, "model_type", ""))


def _architectures(model: Any) -> tuple[str, ...]:
    config = _pretrained_config(model)
    architectures = getattr(config, "architectures", None)
    if architectures is None:
        return ()
    return tuple(str(item) for item in architectures)


def _is_deepseek_family(model_type: str, architectures: tuple[str, ...]) -> bool:
    if model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
        return True
    return any("DeepseekV3" in architecture for architecture in architectures)


def _has_cuda_graph_padding(real_requests: Any, padded_requests: Any) -> bool:
    if real_requests is padded_requests:
        return False
    return (
        real_requests.num_context_requests != padded_requests.num_context_requests
        or real_requests.num_generation_requests
        != padded_requests.num_generation_requests)


def _is_dummy_request(request: Any) -> bool:
    return bool(
        getattr(request, "is_attention_dp_dummy", False)
        or getattr(request, "is_cuda_graph_dummy", False)
        or getattr(request, "is_dummy_request", False))


def _real_generation_request_count(requests: Any) -> int:
    return sum(
        1 for request in getattr(requests, "generation_requests", ())
        if not _is_dummy_request(request))


def _has_draft_tokens(requests: Any) -> bool:
    for request in getattr(requests, "generation_requests", ()):
        if _is_dummy_request(request):
            continue
        draft_tokens = getattr(request, "draft_tokens", None)
        if draft_tokens is None:
            continue
        try:
            if len(draft_tokens) > 0:
                return True
        except TypeError:
            return True
    return False


def build_persistent_decode_plan(
    *,
    real_requests: Any,
    padded_requests: Any,
    attn_metadata: Any,
    inputs: dict[str, Any],
    model: Any,
    kv_cache_manager: Any,
    can_run_graph: bool,
    enable_spec_decode: bool,
    is_draft_model: bool,
) -> PersistentDecodePlan:
    model_type = _model_type(model)
    architectures = _architectures(model)
    metadata_seqs = _as_int(getattr(attn_metadata, "num_seqs", 0))
    metadata_generations = _as_int(getattr(attn_metadata, "num_generations", 0))
    request_ids = _request_ids_from_metadata(attn_metadata, metadata_seqs)
    seq_lens = _seq_lens_from_metadata(attn_metadata, metadata_seqs)
    cached_tokens = _cached_tokens_from_metadata(attn_metadata, metadata_seqs)
    physical_generation_requests = real_requests.num_generation_requests
    real_generation_requests = _real_generation_request_count(real_requests)

    reason = "eligible_decode_only_deepseek_batch"
    eligible = True
    if kv_cache_manager is None:
        eligible = False
        reason = "missing_kv_cache_manager"
    elif real_requests.num_context_requests != 0:
        eligible = False
        reason = "context_requests_present"
    elif physical_generation_requests <= 0:
        eligible = False
        reason = "no_generation_requests"
    elif _has_draft_tokens(real_requests):
        eligible = False
        reason = "draft_tokens_present"
    elif enable_spec_decode:
        eligible = False
        reason = "spec_decode_enabled"
    elif is_draft_model:
        eligible = False
        reason = "draft_model_forward"
    elif not _is_deepseek_family(model_type, architectures):
        eligible = False
        reason = "unsupported_model_family"
    elif _as_int(getattr(attn_metadata, "num_contexts", 0)) != 0:
        eligible = False
        reason = "metadata_contexts_present"
    elif metadata_generations != padded_requests.num_generation_requests:
        eligible = False
        reason = "metadata_generation_mismatch"
    elif seq_lens and any(seq_len != 1 for seq_len in seq_lens):
        eligible = False
        reason = "non_scalar_decode_step"

    return PersistentDecodePlan(
        eligible=eligible,
        reason=reason,
        model_type=model_type,
        architectures=architectures,
        real_context_requests=real_requests.num_context_requests,
        real_generation_requests=real_generation_requests,
        padded_context_requests=padded_requests.num_context_requests,
        padded_generation_requests=padded_requests.num_generation_requests,
        metadata_contexts=_as_int(getattr(attn_metadata, "num_contexts", 0)),
        metadata_generations=metadata_generations,
        metadata_tokens=_as_int(getattr(attn_metadata, "num_tokens", 0)),
        metadata_seqs=metadata_seqs,
        request_ids=request_ids,
        seq_lens=seq_lens,
        cached_tokens=cached_tokens,
        input_tokens=_input_token_count(inputs),
        cuda_graph_replay=can_run_graph,
        cuda_graph_padding=_has_cuda_graph_padding(real_requests, padded_requests),
        enable_spec_decode=enable_spec_decode,
        is_draft_model=is_draft_model,
        kv_cache_manager_type=type(kv_cache_manager).__name__,
        attention_metadata_type=type(attn_metadata).__name__,
    )


def maybe_log_persistent_decode_plan(
    *,
    real_requests: Any,
    padded_requests: Any,
    attn_metadata: Any,
    inputs: dict[str, Any],
    model: Any,
    kv_cache_manager: Any,
    can_run_graph: bool,
    enable_spec_decode: bool,
    is_draft_model: bool,
    force: bool = False,
) -> PersistentDecodePlan | None:
    if not (force or persistent_decode_plan_enabled()):
        return None
    plan = build_persistent_decode_plan(
        real_requests=real_requests,
        padded_requests=padded_requests,
        attn_metadata=attn_metadata,
        inputs=inputs,
        model=model,
        kv_cache_manager=kv_cache_manager,
        can_run_graph=can_run_graph,
        enable_spec_decode=enable_spec_decode,
        is_draft_model=is_draft_model,
    )
    if persistent_decode_plan_debug_enabled():
        logger.info(f"OPTRT_PERSISTENT_DECODE_PLAN {asdict(plan)}")
    return plan

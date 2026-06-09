from dataclasses import dataclass
from enum import IntEnum
import os
from typing import Any, Dict, List, Optional

import numpy as np

# isort: off
# needed before trying to import bindings to load tensorrt_libs
import tensorrt as trt  # noqa
# isort: on

from tensorrt_llm.bindings import executor as tllme


def _optrt_disagg_params_debug_format(value: object) -> str:
    if value is None:
        return "None"
    if hasattr(value, "name"):
        return str(getattr(value, "name"))
    if isinstance(value, (bool, int, float, str)):
        return str(value).replace(" ", "_").replace("\n", "\\n")
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if len(values) <= 8 and all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in values):
            return str(values).replace(" ", "_")
        return f"len:{len(values)}"
    if isinstance(value, dict):
        return f"keys:{','.join(str(key) for key in value.keys())}"
    return str(value).replace(" ", "_").replace("\n", "\\n")


def _optrt_disagg_params_debug_len(value: object) -> str:
    if value is None:
        return "None"
    try:
        return str(len(value))  # type: ignore[arg-type]
    except TypeError:
        return "n/a"


def _optrt_disagg_params_debug_head(value: object, limit: int = 8) -> str:
    if value is None:
        return "None"
    try:
        values = list(value)[:limit]  # type: ignore[arg-type]
    except TypeError:
        return "n/a"
    return _optrt_disagg_params_debug_format(values)


def _optrt_disagg_params_debug(event: str, **fields: object) -> None:
    if os.environ.get("TRTLLM_OPTRT_DISAGG_PARAMS_DEBUG", "0") != "1":
        return
    parts = ["OPTRT_DISAGG_PARAMS_DEBUG", f"event={event}"]
    parts.extend(
        f"{key}={_optrt_disagg_params_debug_format(value)}"
        for key, value in fields.items())
    print(" ".join(parts), flush=True)


class DisaggScheduleStyle(IntEnum):
    CONTEXT_FIRST = 0
    GENERATION_FIRST = 1


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
        draft_tokens (List[int]): The draft tokens of the generation request
        disagg_request_id (int): The disaggregated request id, if set, both context and generation requests will use it
         as underlying request id.
        first_gen_log_probs (List): The logprobs for first_gen_tokens, produced during prefill.
         Each entry is a list (one per beam) of TokenLogprobs (list of dict[int, Logprob]).
        first_gen_logits (List): The generation logits for first_gen_tokens, produced during prefill.
         Each entry is a torch.Tensor of shape [num_tokens, vocab_size] (one per beam/sequence).
        ctx_usage (Dict[str, Any]): The context usage payload to preserve exact
         usage accounting on the generation server.

        multimodal_embedding_handles (List[Dict[str, Any]]): The resulting multimodal embedding handles from ViT.
        multimodal_hashes (List[List[int]]): The multimodal hashes of each multimodal item in the request.
    """

    request_type: Optional[str] = None
    # P-D Disaggregated Params
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None
    # If disagg_request_id is set, both context and generation requests will use it as underlying request id.
    disagg_request_id: Optional[int] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None
    ctx_usage: Optional[Dict[str, Any]] = None

    # E-P Disaggregated Params
    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = (
        None  # multimodal embedding handles should be a list of cudaIPC handles for each mm_embedding
    )
    multimodal_hashes: Optional[List[List[int]]] = (
        None  # user provided mm hashes should be a list of 8 integers
    )
    mrope_position_ids_handle: Optional[Dict[str, Any]] = None
    mrope_position_deltas_handle: Optional[Dict[str, Any]] = None

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        # Prefer disagg_request_id over ctx_request_id
        request_id = (
            self.disagg_request_id if self.disagg_request_id is not None else self.ctx_request_id
        )
        # `first_gen_tokens` is now required by bindings and cannot be None.
        first_gen_tokens = self.first_gen_tokens if self.first_gen_tokens is not None else []
        _optrt_disagg_params_debug(
            "get_context_phase_params",
            request_type=self.request_type,
            request_id=request_id,
            ctx_request_id=self.ctx_request_id,
            disagg_request_id=self.disagg_request_id,
            ctx_dp_rank=self.ctx_dp_rank,
            ctx_info_endpoint=self.ctx_info_endpoint,
            first_gen_tokens_len=_optrt_disagg_params_debug_len(
                first_gen_tokens),
            first_gen_tokens_head=_optrt_disagg_params_debug_head(
                first_gen_tokens),
            draft_tokens_len=_optrt_disagg_params_debug_len(self.draft_tokens),
            draft_tokens_head=_optrt_disagg_params_debug_head(
                self.draft_tokens),
            opaque_state_present=self.opaque_state is not None)
        return tllme.ContextPhaseParams(
            first_gen_tokens,
            request_id,
            self.opaque_state,
            self.draft_tokens,
            self.ctx_dp_rank,
            self.ctx_info_endpoint,
        )

    def get_request_type(self) -> tllme.RequestType:
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )

    def __post_init__(self):
        if self.request_type is not None:
            self.request_type = self.request_type.lower()
            if self.request_type not in [
                "context_only",
                "generation_only",
                "context_and_generation",
            ]:
                raise ValueError(
                    f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                    "context_and_generation"
                )
        if self.multimodal_embedding_handles is not None:
            if self.multimodal_hashes is not None:
                # if mm hashes are provided, kvcache reuse can be enabled
                assert len(self.multimodal_embedding_handles) == len(self.multimodal_hashes), (
                    "multimodal_embedding_handles and multimodal_hashes must have the same length"
                )
                for mm_hash in self.multimodal_hashes:
                    assert isinstance(mm_hash, list), "mm_hash must be a list"
                    assert len(mm_hash) == 8, "mm_hash must be a list of 8 integers"
                    assert all(isinstance(x, int) for x in mm_hash), "mm_hash must contain integers"
            else:
                # if user did not provide mm embedding handles, kvcache reuse will be disabled
                assert len(self.multimodal_embedding_handles) > 0, (
                    "multimodal_embedding_handles must be provided"
                )
                vals = np.random.randint(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=8, dtype=np.int32
                ).tolist()
                self.multimodal_hashes = [vals] * len(self.multimodal_embedding_handles)

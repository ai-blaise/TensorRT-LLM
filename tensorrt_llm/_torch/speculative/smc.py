import os
from dataclasses import dataclass

import torch

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import (BaseResourceManager, ResourceManager,
                                           ResourceManagerType)
from ..pyexecutor.sampler import (DEFAULT_BEAM_IDX, FinishReasonsList,
                                  SampleState, TorchSampler, add_token)
from .interface import SpecMetadata
from .model_drafter import ModelDrafter
from .spec_tree_manager import SpecTreeManager


def _nonempty_ctx_endpoint(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return any(_nonempty_ctx_endpoint(item) for item in value)
    return True


def _smc_request_pin_required(request: LlmRequest) -> bool:
    params = getattr(request, "py_disaggregated_params", None)
    request_type = getattr(params, "request_type", None)
    return bool(getattr(request, "is_generation_only_request", lambda: False)()
                or request_type == "generation_only")


def validate_smc_decode_request_pin(request: LlmRequest) -> tuple[object, object, object]:
    """Fail closed when SMC-SD decode lacks disagg request pin metadata.

    The Moondream overlap path commits draft tokens one iteration after the
    draft forward. In disaggregated generation-only decode, that delayed commit
    must stay attached to the same prefill KV handoff, so require the request id,
    ctx DP rank, and ctx endpoint before consuming SMC draft payloads.
    """
    if os.environ.get("TRTLLM_SMC_REQUIRE_REQUEST_PIN", "1") == "0":
        return None, None, None
    if not _smc_request_pin_required(request):
        return None, None, None

    params = getattr(request, "py_disaggregated_params", None)
    if params is None:
        raise RuntimeError(
            "SMC-SD decode requires disaggregated request pin metadata "
            "for generation-only requests")

    disagg_request_id = getattr(params, "disagg_request_id", None)
    if disagg_request_id is None:
        disagg_request_id = getattr(params, "ctx_request_id", None)
    ctx_dp_rank = getattr(params, "ctx_dp_rank", None)
    ctx_info_endpoint = getattr(params, "ctx_info_endpoint", None)

    missing = []
    if disagg_request_id is None:
        missing.append("disagg_request_id")
    if ctx_dp_rank is None:
        missing.append("ctx_dp_rank")
    if not _nonempty_ctx_endpoint(ctx_info_endpoint):
        missing.append("ctx_info_endpoint")
    if missing:
        raise RuntimeError(
            "SMC-SD decode requires request pin metadata; missing "
            + ", ".join(missing))
    return disagg_request_id, ctx_dp_rank, ctx_info_endpoint


def build_smc_particle_choices(gamma: int, n_particles: int) -> list[list[int]]:
    choices: list[list[int]] = []
    for depth in range(1, gamma + 1):
        for particle_idx in range(n_particles):
            choices.append([particle_idx] + [0] * (depth - 1))
    return choices


def _smc_debug_scalar(value: object) -> object:
    if torch.is_tensor(value):
        if value.device.type == "cpu" and value.numel() == 1:
            return int(value.item())
        return f"tensor:{tuple(value.shape)}:{value.device.type}"
    if hasattr(value, "name"):
        return getattr(value, "name")
    return value


def _smc_debug_format(value: object) -> str:
    value = _smc_debug_scalar(value)
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float, str)):
        return str(value).replace(" ", "_").replace("\n", "\\n")
    if isinstance(value, (list, tuple, set)):
        values = [_smc_debug_scalar(item) for item in list(value)]
        if len(values) <= 8 and all(
                isinstance(item, (bool, int, float, str)) or item is None
                for item in values):
            return str(values).replace(" ", "_")
        return f"len:{len(values)}"
    if isinstance(value, dict):
        return f"keys:{','.join(str(key) for key in value.keys())}"
    return str(value).replace(" ", "_").replace("\n", "\\n")


def _smc_debug_len(value: object) -> str:
    if value is None:
        return "None"
    try:
        return str(len(value))  # type: ignore[arg-type]
    except TypeError:
        return "n/a"


def _smc_debug_head(value: object, limit: int = 8) -> str:
    if value is None:
        return "None"
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            return f"tensor:{tuple(value.shape)}:{value.device.type}"
        value = value.reshape(-1)[:limit].tolist()
    try:
        values = list(value)[:limit]  # type: ignore[arg-type]
    except TypeError:
        return _smc_debug_format(value)
    return _smc_debug_format(values)


def _smc_debug_shape(value: object) -> str:
    if value is None:
        return "None"
    shape = getattr(value, "shape", None)
    if shape is None:
        return "n/a"
    return _smc_debug_format(tuple(shape))


def _smc_debug(event: str,
               request: LlmRequest | None = None,
               **fields: object) -> None:
    if os.environ.get("TRTLLM_OPTRT_SMC_DEBUG", "0") != "1":
        return
    parts = ["OPTRT_SMC_DEBUG", f"event={event}"]
    if request is not None:
        parts.extend([
            f"request_id={_smc_debug_format(getattr(request, 'request_id', None))}",
            f"py_request_id={_smc_debug_format(getattr(request, 'py_request_id', None))}",
            f"state={_smc_debug_format(getattr(request, 'state', None))}",
            f"seq_slot={_smc_debug_format(getattr(request, 'seq_slot', None))}",
            f"py_seq_slot={_smc_debug_format(getattr(request, 'py_seq_slot', None))}",
        ])
        draft_tokens = getattr(request, "py_draft_tokens", None)
        last_draft_tokens = getattr(request, "py_last_draft_tokens", None)
        parts.extend([
            f"py_draft_tokens_len={_smc_debug_len(draft_tokens)}",
            f"py_draft_tokens_head={_smc_debug_head(draft_tokens)}",
            f"py_last_draft_tokens_len={_smc_debug_len(last_draft_tokens)}",
            f"py_last_draft_tokens_head={_smc_debug_head(last_draft_tokens)}",
            "py_smc_draft_token_log_probs_shape="
            f"{_smc_debug_shape(getattr(request, 'py_smc_draft_token_log_probs', None))}",
            f"py_target_probs_shape={_smc_debug_shape(getattr(request, 'py_target_probs', None))}",
        ])
        disagg_params = getattr(request, "py_disaggregated_params", None)
        if disagg_params is not None:
            parts.extend([
                "disagg_request_type="
                f"{_smc_debug_format(getattr(disagg_params, 'request_type', None))}",
                "disagg_request_id="
                f"{_smc_debug_format(getattr(disagg_params, 'disagg_request_id', None))}",
                "disagg_ctx_request_id="
                f"{_smc_debug_format(getattr(disagg_params, 'ctx_request_id', None))}",
                "disagg_ctx_dp_rank="
                f"{_smc_debug_format(getattr(disagg_params, 'ctx_dp_rank', None))}",
                "disagg_ctx_info_endpoint="
                f"{_smc_debug_format(getattr(disagg_params, 'ctx_info_endpoint', None))}",
                "disagg_draft_tokens_len="
                f"{_smc_debug_len(getattr(disagg_params, 'draft_tokens', None))}",
                "disagg_draft_tokens_head="
                f"{_smc_debug_head(getattr(disagg_params, 'draft_tokens', None))}",
            ])
    parts.extend(
        f"{key}={_smc_debug_format(value)}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


@dataclass
class SMCSpecMetadata(SpecMetadata):
    """Runtime metadata for two-model SMC-SD speculative decoding."""

    gamma: int = 6
    n_particles: int = 4
    resample_threshold: float = 0.5
    target_temperature: float = 1.0
    draft_temperature: float = 1.0
    particle_log_weights: torch.Tensor | None = None
    particle_ess: torch.Tensor | None = None
    resample_indices: torch.Tensor | None = None
    accepted_lengths: torch.Tensor | None = None
    smc_resource_manager: "SMCResourceManager | None" = None

    def __post_init__(self):
        super().__post_init__()
        self.is_spec_dec_tree = True
        self.is_spec_dec_dynamic_tree = False
        self.particle_log_weights = torch.zeros(
            (self.max_num_requests, self.n_particles),
            dtype=torch.float32,
            device="cuda",
        )
        self.particle_ess = torch.empty(
            (self.max_num_requests,),
            dtype=torch.float32,
            device="cuda",
        )
        self.resample_indices = torch.empty(
            (self.max_num_requests, self.n_particles),
            dtype=torch.int32,
            device="cuda",
        )
        self.accepted_lengths = torch.empty(
            (self.max_num_requests,),
            dtype=torch.int32,
            device="cuda",
        )

    def prepare(self):
        super().prepare()
        self.runtime_draft_len = self.max_total_draft_tokens


class SMCResourceManager(BaseResourceManager):
    """Tracks SMC-SD particle state across decode iterations."""

    def __init__(self, spec_config, max_num_requests: int):
        self.gamma = spec_config.gamma
        self.n_particles = spec_config.n_particles
        self.resample_threshold = spec_config.resample_threshold
        self.target_temperature = spec_config.target_temperature
        self.draft_temperature = spec_config.draft_temperature
        self.max_num_requests = max_num_requests
        self.is_first_draft = True
        self.log_weights: dict[int, torch.Tensor] = {}
        self.step_counts: dict[int, int] = {}
        self.accepted_lengths: dict[int, int] = {}
        self.selected_particles: dict[int, int] = {}
        self.particle_choices = build_smc_particle_choices(
            self.gamma, self.n_particles)
        self.spec_tree_manager = SpecTreeManager(
            max_num_requests=max_num_requests,
            use_dynamic_tree=False,
            max_total_draft_tokens=len(self.particle_choices),
            max_draft_len=self.gamma,
            eagle_choices=self.particle_choices,
            dynamic_tree_max_topK=0,
        )

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def add_dummy_requests(self, request_ids: list[int]):
        for request_id in request_ids:
            self._ensure_request(int(request_id))

    def prepare_resources(self, scheduled_batch):
        for request in scheduled_batch.all_requests():
            self._ensure_request(int(request.py_request_id))

    def update_resources(self, scheduled_batch):
        for request in scheduled_batch.all_requests():
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                self.free_resources(request)

    def free_resources(self, request: LlmRequest):
        request_id = int(request.py_request_id)
        self.log_weights.pop(request_id, None)
        self.step_counts.pop(request_id, None)
        self.accepted_lengths.pop(request_id, None)
        self.selected_particles.pop(request_id, None)

    def reset_request(self, request_id: int) -> None:
        self.log_weights[request_id] = torch.zeros(
            (self.n_particles,), dtype=torch.float32, device="cuda")
        self.step_counts[request_id] = 0
        self.accepted_lengths[request_id] = 0
        self.selected_particles[request_id] = 0

    def record_acceptance(self, request_id: int, accepted_length: int) -> None:
        self._ensure_request(request_id)
        self.accepted_lengths[request_id] = accepted_length
        self.step_counts[request_id] += 1

    def record_logprob_diff(
        self, request_id: int, particle_idx: int, logprob_diff: torch.Tensor
    ) -> None:
        self._ensure_request(request_id)
        self.log_weights[request_id][particle_idx] += logprob_diff.to(
            dtype=torch.float32, device=self.log_weights[request_id].device
        )

    def select_particle(
        self, request_id: int, logprob_diffs: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        self._ensure_request(request_id)
        log_weights = self.log_weights[request_id]
        log_weights.add_(logprob_diffs.to(device=log_weights.device,
                                          dtype=torch.float32))
        weights = torch.softmax(log_weights, dim=0)
        ess = torch.reciprocal(torch.sum(weights * weights))
        selected_particle = int(torch.argmax(weights).item())
        self.selected_particles[request_id] = selected_particle
        if bool((ess < self.n_particles * self.resample_threshold).item()):
            log_weights.zero_()
            log_weights[selected_particle] = torch.log(
                weights[selected_particle].clamp_min(1e-30))
        return selected_particle, ess

    def select_particles_batched(
        self, request_ids: list[int], logprob_diffs: torch.Tensor
    ) -> tuple[list[int], torch.Tensor]:
        """Vectorized batch equivalent of ``select_particle``.

        Folds the whole decode batch into a single GPU computation plus a
        single device->host transfer, replacing the per-request
        ``argmax(...).item()`` + ``(ess < thr).item()`` syncs (3 per request)
        that otherwise serialize the verification loop and starve the decode
        pipeline under the overlap scheduler.

        ``logprob_diffs`` is ``[B, n_particles]`` (row i for ``request_ids[i]``)
        and lives on device. State mutation of ``log_weights[gid]`` matches
        ``select_particle`` exactly. Returns host-resident selected particles
        and the device-resident per-row ESS.
        """
        if not request_ids:
            empty = torch.empty((0,), dtype=torch.float32, device="cuda")
            return [], empty
        for request_id in request_ids:
            self._ensure_request(request_id)
        # Gather accumulated weights into a contiguous [B, P] tensor, add this
        # step's diffs, then write the accumulated rows back so per-group state
        # stays in lockstep with the per-request path.
        device = self.log_weights[request_ids[0]].device
        log_weights = torch.stack(
            [self.log_weights[request_id] for request_id in request_ids], dim=0)
        log_weights.add_(logprob_diffs.to(device=device, dtype=torch.float32))
        weights = torch.softmax(log_weights, dim=1)
        ess = torch.reciprocal((weights * weights).sum(dim=1))
        selected = torch.argmax(weights, dim=1)
        resample = ess < (self.n_particles * self.resample_threshold)
        # Resample-reset rows: zero everything except the selected slot, which
        # holds log(weight[selected]); identical to the scalar branch.
        selected_weight = weights.gather(1, selected.unsqueeze(1)).squeeze(1)
        reset_rows = torch.zeros_like(log_weights)
        reset_rows.scatter_(
            1, selected.unsqueeze(1),
            torch.log(selected_weight.clamp_min(1e-30)).unsqueeze(1))
        log_weights = torch.where(resample.unsqueeze(1), reset_rows,
                                  log_weights)
        # Single d2h: all selected-particle decisions at once.
        selected_host = selected.tolist()
        for request_id, particle, row in zip(request_ids, selected_host,
                                             log_weights):
            self.log_weights[request_id].copy_(row)
            self.selected_particles[request_id] = particle
        return selected_host, ess

    def effective_sample_size(self, request_id: int) -> torch.Tensor:
        self._ensure_request(request_id)
        weights = torch.softmax(self.log_weights[request_id], dim=0)
        return torch.reciprocal(torch.sum(weights * weights))

    def needs_resample(self, request_id: int) -> torch.Tensor:
        ess = self.effective_sample_size(request_id)
        return ess < (self.n_particles * self.resample_threshold)

    def _ensure_request(self, request_id: int) -> None:
        if request_id not in self.log_weights:
            self.reset_request(request_id)


class SMCModelDrafter(ModelDrafter):
    """Two-model drafter for SMC-SD."""

    def _pack_static_draft_outputs_for_overlap(
        self,
        outputs: dict[str, torch.Tensor],
        sample_state: SampleState,
    ) -> dict[str, torch.Tensor | SampleState]:
        """Carry the SMC draft-logprob payload through the delayed commit.

        The generic static-draft overlap path stores ``draft_logits`` plus an
        event-gated host token copy. SMCStaticParticleDraftingLoopWrapper returns
        per-token draft log-probs instead of logits, so preserve that payload and
        let process_static_draft_outputs consume the same pinned/evented token
        copy used by the Moondream-style ping-pong pipeline.
        """
        return {
            "new_draft_tokens": outputs["new_draft_tokens"],
            "draft_token_log_probs": outputs["draft_token_log_probs"],
            "sample_state": sample_state,
        }

    def process_static_draft_outputs(self, outputs, draft_batch) -> None:
        if not isinstance(outputs, dict) or "draft_token_log_probs" not in outputs:
            _smc_debug(
                "drafter_process_static_passthrough",
                outputs_type=type(outputs).__name__,
                has_draft_token_log_probs=isinstance(outputs, dict)
                and "draft_token_log_probs" in outputs)
            super().process_static_draft_outputs(outputs, draft_batch)
            return

        sample_state = outputs.get("sample_state")
        _smc_debug(
            "drafter_process_static_enter",
            sample_state_present=sample_state is not None,
            new_draft_tokens_shape=_smc_debug_shape(
                outputs.get("new_draft_tokens")),
            draft_token_log_probs_shape=_smc_debug_shape(
                outputs.get("draft_token_log_probs")),
            draft_batch_size=len(draft_batch.all_requests()))
        if sample_state is not None:
            sample_state.sampler_event.synchronize()
            draft_tokens_host = sample_state.host.new_tokens
            used_pinned_host_tokens = True
            _smc_debug(
                "drafter_process_static_sample_state_ready",
                host_new_tokens_shape=_smc_debug_shape(draft_tokens_host),
                host_new_tokens_head=_smc_debug_head(draft_tokens_host))
        else:
            if os.environ.get("TRTLLM_SMC_ALLOW_UNPINNED_DRAFT_COMMIT", "0") != "1":
                raise RuntimeError(
                    "SMC-SD Moondream decode requires evented pinned host "
                    "draft tokens; set TRTLLM_SMC_ALLOW_UNPINNED_DRAFT_COMMIT=1 "
                    "only for explicit diagnostic fallback")
            draft_tokens_host = outputs["new_draft_tokens"].cpu()
            used_pinned_host_tokens = False
            _smc_debug(
                "drafter_process_static_unpinned_fallback",
                host_new_tokens_shape=_smc_debug_shape(draft_tokens_host),
                host_new_tokens_head=_smc_debug_head(draft_tokens_host))
        draft_token_log_probs = outputs["draft_token_log_probs"]

        for req_idx, req in enumerate(draft_batch.all_requests()):
            target_model_req = self.req_id_to_old_request[req.py_request_id]
            _smc_debug(
                "drafter_process_static_candidate",
                target_model_req,
                draft_request_id=req.py_request_id,
                req_idx=req_idx,
                used_pinned_host_tokens=used_pinned_host_tokens)
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                _smc_debug(
                    "drafter_process_static_skip_non_generation",
                    target_model_req,
                    draft_request_id=req.py_request_id,
                    req_idx=req_idx)
                continue
            disagg_request_id, ctx_dp_rank, ctx_info_endpoint = (
                validate_smc_decode_request_pin(target_model_req))
            # This handoff trace runs per request per draft-commit on the SMC
            # decode hot path. At INFO level (usually enabled in prod) it built
            # and emitted a 5-field f-string every step -- pure host overhead
            # under the overlap scheduler. Gate behind TRTLLM_OPTRT_SMC_DEBUG so
            # the disagg pin trace stays available when debugging but costs
            # nothing in steady-state decode. The pin VALIDATION above is
            # unchanged (still fail-closed every step); only the log is gated.
            if os.environ.get("TRTLLM_OPTRT_SMC_DEBUG", "0") == "1":
                logger.info(
                    "SMC Moondream decode handoff preserved "
                    "draft_token_log_probs sample_state.sampler_event "
                    f"pinned_host_tokens={used_pinned_host_tokens} "
                    f"request_id={target_model_req.py_request_id} "
                    f"disagg_request_id={disagg_request_id} "
                    f"ctx_dp_rank={ctx_dp_rank} "
                    f"ctx_info_endpoint={ctx_info_endpoint}")
            _smc_debug(
                "drafter_process_static_commit_start",
                target_model_req,
                draft_request_id=req.py_request_id,
                req_idx=req_idx,
                disagg_request_id=disagg_request_id,
                ctx_dp_rank=ctx_dp_rank,
                ctx_info_endpoint=ctx_info_endpoint,
                max_total_draft_tokens=self.max_total_draft_tokens,
                token_log_probs_shape=_smc_debug_shape(draft_token_log_probs))
            target_model_req.py_draft_tokens = []
            token_log_probs = []
            for token_idx in range(self.max_total_draft_tokens):
                target_model_req.py_draft_tokens.append(
                    draft_tokens_host[token_idx][req_idx])
                token_log_probs.append(draft_token_log_probs[token_idx][req_idx])

            target_model_req.py_draft_logits = None
            target_model_req.py_smc_draft_token_log_probs = torch.stack(
                token_log_probs)
            _smc_debug(
                "drafter_process_static_commit_done",
                target_model_req,
                draft_request_id=req.py_request_id,
                req_idx=req_idx,
                committed_token_log_probs_shape=_smc_debug_shape(
                    target_model_req.py_smc_draft_token_log_probs))


class SMCSampler(TorchSampler):
    """Sampler for SMC-SD target verification.

    Each iteration verifies ``gamma`` draft positions plus one target bonus
    position using the PyTorch runtime's draft/target verification path.
    """

    def __init__(self, args: TorchSampler.Args, *, gamma: int,
                 n_particles: int, resample_threshold: float,
                 draft_temperature: float):
        super().__init__(args)
        self.gamma = gamma
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.draft_temperature = draft_temperature
        self._particle_token_indices = [[depth * n_particles + particle
                                         for depth in range(gamma)]
                                        for particle in range(n_particles)]
        # Host-side static parent-step layout, used by the accept hot path
        # instead of a per-request device->host .tolist(). Built lazily and
        # cached by _get_parent_steps_host; prime it here for the normal path.
        self._get_parent_steps_host()
        self._particle_index_cache: dict[torch.device,
                                         tuple[torch.Tensor, torch.Tensor]] = {}
        # Per-step cache of batched particle selection, keyed by SMC group id.
        # Populated by the pre-pass in update_requests so the per-request
        # process_draft_tokens path consumes host scalars without per-request
        # device->host syncs. Values: (selected_particle:int, ess:Tensor).
        self._batched_selection: dict[int, tuple[int, torch.Tensor]] = {}

    def should_provide_draft_probs(self, request) -> bool:
        return True

    def _can_use_fast_greedy_path(self, requests) -> bool:
        return False

    def update_requests(self, state, resource_manager=None) -> None:
        """Run a batched particle-selection pre-pass, then defer to the base
        per-request update loop.

        The base loop calls ``process_draft_tokens`` once per request; for SMC
        that otherwise costs three device->host syncs per request
        (argmax(logprob_diffs), argmax(weights), ess<thr). The pre-pass folds
        the whole decode batch into one vectorized GPU computation plus a single
        transfer; ``process_draft_tokens`` then reads cached host scalars.
        """
        self._batched_selection = {}
        spec_manager = None
        if resource_manager is not None:
            spec_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if isinstance(spec_manager, SMCResourceManager) and state.requests:
            group_ids: list[int] = []
            diff_rows: list[torch.Tensor] = []
            _smc_debug(
                "sampler_update_enter",
                request_count=len(state.requests),
                spec_manager=type(spec_manager).__name__)
            for request in state.requests:
                if request.state == LlmRequestState.GENERATION_COMPLETE:
                    _smc_debug("sampler_update_skip_generation_complete",
                               request)
                    continue
                draft_tokens = getattr(request, "py_draft_tokens", None)
                if not draft_tokens:
                    _smc_debug("sampler_update_skip_no_draft_tokens", request)
                    continue
                if getattr(request, "py_target_probs", None) is None:
                    _smc_debug("sampler_update_skip_no_target_probs", request)
                    continue
                if getattr(request, "py_smc_draft_token_log_probs",
                           None) is None:
                    _smc_debug("sampler_update_skip_no_smc_log_probs", request)
                    continue
                group_ids.append(
                    int(getattr(request, "py_smc_group_id",
                                request.py_request_id)))
                diff_rows.append(self._compute_particle_logprob_diffs(request))
            if group_ids:
                logprob_diffs = torch.stack(diff_rows, dim=0)
                _smc_debug(
                    "sampler_update_select_particles_start",
                    group_ids=group_ids,
                    logprob_diffs_shape=_smc_debug_shape(logprob_diffs))
                selected, ess = spec_manager.select_particles_batched(
                    group_ids, logprob_diffs)
                for idx, group_id in enumerate(group_ids):
                    self._batched_selection[group_id] = (selected[idx],
                                                         ess[idx])
                    _smc_debug(
                        "sampler_update_select_particles_done",
                        group_id=group_id,
                        selected_particle=selected[idx],
                        ess=ess[idx])
            else:
                _smc_debug("sampler_update_no_selectable_requests")
        super().update_requests(state, resource_manager)
        self._batched_selection = {}

    def process_draft_tokens(
        self,
        request,
        new_tokens_tensor: torch.Tensor,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
        resource_manager: ResourceManager | None = None,
    ) -> int:
        draft_tokens = request.py_draft_tokens
        if draft_tokens is None or len(draft_tokens) == 0:
            _smc_debug("sampler_process_empty_draft_tokens", request)
            return self._process_draft_tokens_greedy(
                request, new_tokens=new_tokens_list, finish_reasons=finish_reasons)

        if getattr(request, "py_target_probs", None) is None:
            _smc_debug("sampler_process_missing_target_probs", request)
            raise RuntimeError("SMC-SD requires target token probabilities.")
        if getattr(request, "py_smc_draft_token_log_probs", None) is None:
            _smc_debug("sampler_process_missing_smc_log_probs", request)
            raise RuntimeError(
                "SMC-SD requires selected draft token log probabilities.")

        group_id = int(getattr(request, "py_smc_group_id",
                               request.py_request_id))
        cached = getattr(self, "_batched_selection", {}).get(group_id)
        if cached is not None:
            # Batched pre-pass already advanced particle state and resolved the
            # selection on device; consume the cached host scalar (no sync).
            selected_particle, ess = cached
            setattr(request, "py_smc_effective_sample_size", ess)
            setattr(request, "py_smc_selected_particle", selected_particle)
            _smc_debug(
                "sampler_process_cached_selection",
                request,
                group_id=group_id,
                selected_particle=selected_particle,
                ess=ess)
        else:
            # Fallback (e.g. no SMC resource manager): per-request path.
            logprob_diffs = self._compute_particle_logprob_diffs(request)
            selected_particle = int(torch.argmax(logprob_diffs).item())
            _smc_debug(
                "sampler_process_fallback_selection_start",
                request,
                group_id=group_id,
                selected_particle=selected_particle,
                logprob_diffs_shape=_smc_debug_shape(logprob_diffs))
            if resource_manager is not None:
                spec_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.SPEC_RESOURCE_MANAGER)
                if isinstance(spec_manager, SMCResourceManager):
                    selected_particle, ess = spec_manager.select_particle(
                        group_id, logprob_diffs)
                    setattr(request, "py_smc_effective_sample_size", ess)
                    setattr(request, "py_smc_selected_particle",
                            selected_particle)
                    _smc_debug(
                        "sampler_process_fallback_selection_done",
                        request,
                        group_id=group_id,
                        selected_particle=selected_particle,
                        ess=ess)

        num_accepted = self._accept_selected_particle(
            request=request,
            selected_particle=selected_particle,
            new_tokens_tensor=new_tokens_tensor,
            new_tokens_list=new_tokens_list,
            finish_reasons=finish_reasons,
        )
        if resource_manager is not None:
            spec_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if isinstance(spec_manager, SMCResourceManager):
                group_id = int(getattr(
                    request, "py_smc_group_id", request.py_request_id))
                spec_manager.record_acceptance(group_id, num_accepted)
        _smc_debug(
            "sampler_process_done",
            request,
            group_id=group_id,
            selected_particle=selected_particle,
            num_accepted=num_accepted)
        return num_accepted

    def _get_particle_index_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index_cache = getattr(self, "_particle_index_cache", None)
        if index_cache is None:
            index_cache = {}
            self._particle_index_cache = index_cache
        if device not in index_cache:
            particle_token_indices = torch.tensor(
                self._particle_token_indices, dtype=torch.long, device=device)
            parent_steps = torch.empty_like(particle_token_indices)
            parent_steps[:, 0] = 0
            if self.gamma > 1:
                parent_steps[:, 1:] = particle_token_indices[:, :-1] + 1
            index_cache[device] = (particle_token_indices, parent_steps)
        return index_cache[device]

    def _get_parent_steps_host(self) -> list[list[int]]:
        """Host-side parent-step layout per particle, lazily cached.

        Mirrors the device ``parent_steps`` from ``_get_particle_index_tensors``
        but is purely static config (a function of gamma / n_particles). Reading
        it from host avoids a per-request device->host ``.tolist()`` of the
        cached device tensor in the accept hot path. ``getattr`` fallback so any
        construction path (incl. test factories using ``object.__new__``) works.
        """
        cached = getattr(self, "_parent_steps_host", None)
        if cached is None:
            cached = [[0] + [row[d - 1] + 1 for d in range(1, self.gamma)]
                      for row in self._particle_token_indices]
            self._parent_steps_host = cached
        return cached

    def _compute_particle_logprob_diffs(self, request) -> torch.Tensor:
        target_probs = request.py_target_probs
        draft_log_probs = request.py_smc_draft_token_log_probs
        device = target_probs.device
        particle_token_indices, parent_steps = self._get_particle_index_tensors(
            device)
        draft_token_ids = torch.tensor(
            [int(token) for token in request.py_draft_tokens],
            dtype=torch.long,
            device=device,
        )[particle_token_indices]
        target_token_probs = target_probs[parent_steps.reshape(-1)].gather(
            1, draft_token_ids.reshape(-1, 1)).reshape(self.n_particles,
                                                       self.gamma)
        selected_draft_log_probs = draft_log_probs.to(
            device=device, dtype=torch.float32)[particle_token_indices]
        return (torch.log(target_token_probs.clamp_min(1e-30)) -
                selected_draft_log_probs).sum(dim=1)

    def _accept_selected_particle(
        self,
        request,
        selected_particle: int,
        new_tokens_tensor: torch.Tensor,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
    ) -> int:
        """Verify the selected particle's draft chain against the target.

        SMC selects the highest-importance-weight particle (trajectory); the
        tokens along that trajectory are still only PROPOSALS and must be
        verified against the target before being emitted.  We walk the selected
        chain depth-by-depth: at depth ``d`` the target's own token (the one the
        target model already sampled at the parent tree node, respecting
        ``target_temperature``) lives at
        ``new_tokens_tensor[parent_step_d, seq_slot]``, where ``parent_step_d``
        follows the same ``parent_steps`` convention that
        ``_compute_particle_logprob_diffs`` uses to read ``py_target_probs``.
        Accept the draft token iff it equals that target token; stop at the first
        mismatch and emit the target's own token there.  This is the canonical
        speculative-decoding accept (cf. ``_process_draft_tokens_greedy`` and
        ``_process_draft_tokens_tree``), walking one tree path instead of a flat
        list, and is correct for any target sampling temperature because it
        verifies against what the target ACTUALLY sampled, not a recomputed
        argmax.

        This is the correctness fix: the previous implementation accepted the
        whole chain unconditionally (acceptance_length == gamma+1 every step), so
        the weak GLM draft's tokens were emitted verbatim -> token salad.  The
        per-token importance weights from ``_compute_particle_logprob_diffs``
        still drive WHICH particle is selected, preserving SMC's acceptance-rate
        benefit; this method only gates emission on target agreement.
        """
        token_indices = self._particle_token_indices[selected_particle]
        seq_slot = request.py_seq_slot
        assert seq_slot is not None

        # parent_steps[selected, d] = tree node whose target sample validates the
        # depth-d draft token (root==0 for d==0). This is static config, so read
        # the precomputed host list instead of a per-call device->host .tolist()
        # of the cached device tensor (this method runs once per request per step).
        parent_steps_sel = self._get_parent_steps_host()[selected_particle]
        num_nodes = int(request.py_target_probs.shape[0])
        # Target tokens the model already sampled at every tree node (host).
        target_tokens = new_tokens_tensor[:num_nodes, seq_slot,
                                          DEFAULT_BEAM_IDX].tolist()
        _smc_debug(
            "accept_selected_enter",
            request,
            selected_particle=selected_particle,
            token_indices=token_indices,
            parent_steps=parent_steps_sel,
            num_nodes=num_nodes,
            target_tokens_head=_smc_debug_head(target_tokens),
            new_tokens_tensor_shape=_smc_debug_shape(new_tokens_tensor))

        # Rejection-sampling acceptance (default): accept draft token t at depth d
        # with probability min(1, q/p) where q = P_target(t | parent) and
        # p = P_draft(t).  This is the speculative-sampling criterion (cf.
        # get_rejected_indices) and yields HIGHER acceptance length than exact
        # token match while staying distributionally correct -- crucial under
        # target_temperature>0 (here 1.0), where the target's single sampled
        # token rarely equals the draft's even when the draft is good.  Set
        # SMC_REJECTION_ACCEPT=0 to fall back to exact-match verification.
        use_rejection = os.environ.get("SMC_REJECTION_ACCEPT", "1") != "0"
        target_q = None
        draft_p = None
        rand_u = None
        if use_rejection:
            tp = request.py_target_probs
            dlp = getattr(request, "py_smc_draft_token_log_probs", None)
            if dlp is not None:
                # Per-depth target prob of the draft token and draft prob, on host.
                sel_idx = torch.tensor(token_indices, dtype=torch.long,
                                       device=tp.device)
                parents = torch.tensor(parent_steps_sel, dtype=torch.long,
                                       device=tp.device).clamp_(max=num_nodes - 1)
                draft_ids = torch.tensor(
                    [int(request.py_draft_tokens[i]) for i in token_indices],
                    dtype=torch.long, device=tp.device)
                q = tp[parents].gather(1, draft_ids.unsqueeze(1)).squeeze(1)
                p = dlp.to(device=tp.device,
                           dtype=torch.float32)[sel_idx].exp()
                gen = self.get_generator(tp.device)
                u = torch.rand(len(token_indices), generator=gen,
                               device=tp.device)
                # Fuse the q / p / u read-back into ONE device->host transfer
                # (was three separate .tolist() syncs per request per step).
                target_q, draft_p, rand_u = torch.stack(
                    (q, p.to(q.dtype), u.to(q.dtype)), dim=0).tolist()
            else:
                use_rejection = False
                _smc_debug("accept_selected_disable_rejection_no_log_probs",
                           request)

        if os.environ.get("SMC_ACCEPT_DEBUG") == "1" and not getattr(
                SMCSampler, "_accept_dbg_done", False):
            SMCSampler._accept_dbg_done = True
            draft_flat = [int(t) for t in request.py_draft_tokens]
            tgt_argmax = torch.argmax(request.py_target_probs,
                                      dim=-1).tolist()
            print(f"[SMC_ACCEPT_DEBUG] sel={selected_particle} "
                  f"n_nodes={num_nodes} gamma={self.gamma} "
                  f"np={self.n_particles} rejection={use_rejection}", flush=True)
            print(f"[SMC_ACCEPT_DEBUG] draft_flat={draft_flat}", flush=True)
            print(f"[SMC_ACCEPT_DEBUG] target_sampled={target_tokens}",
                  flush=True)
            print(f"[SMC_ACCEPT_DEBUG] target_argmax={tgt_argmax}", flush=True)
            print(f"[SMC_ACCEPT_DEBUG] sel_token_indices={token_indices} "
                  f"sel_parent_steps={parent_steps_sel}", flush=True)
            if target_q is not None:
                print(f"[SMC_ACCEPT_DEBUG] target_q={[round(x,4) for x in target_q]} "
                      f"draft_p={[round(x,4) for x in draft_p]}", flush=True)

        num_accepted = 0
        accepted_node_indices: list[int] = []
        for depth, token_idx in enumerate(token_indices):
            parent = parent_steps_sel[depth]
            # Defensive bound: a malformed parent index means we cannot verify
            # this depth -> stop accepting here (correctness over length).
            if parent >= num_nodes:
                _smc_debug(
                    "accept_selected_parent_out_of_bounds",
                    request,
                    selected_particle=selected_particle,
                    depth=depth,
                    parent=parent,
                    num_nodes=num_nodes,
                    num_accepted=num_accepted)
                break
            draft_token = int(request.py_draft_tokens[token_idx])
            target_token = int(target_tokens[parent])
            if use_rejection:
                p = draft_p[depth]
                q = target_q[depth]
                accept_prob = 1.0 if p <= 0.0 else min(1.0, q / p)
                accepted = rand_u[depth] < accept_prob
            else:
                accepted = (draft_token == target_token)
            if not accepted:
                # Reject: emit the target's own sampled token at this position
                # and stop.  (Residual-distribution resampling would need the
                # full draft distribution, which is not retained; emitting the
                # target's sample is the standard fallback and keeps output
                # coherent and target-distributed.)
                new_tokens_tensor[num_accepted, seq_slot,
                                  DEFAULT_BEAM_IDX] = target_token
                request.add_new_token(target_token, DEFAULT_BEAM_IDX)
                self._handle_stop_criteria(request, target_token,
                                           beam_idx=DEFAULT_BEAM_IDX,
                                           max_seq_len=self.max_seq_len)
                request.py_num_accepted_draft_tokens_indices = (
                    accepted_node_indices)
                _smc_debug(
                    "accept_selected_reject",
                    request,
                    selected_particle=selected_particle,
                    depth=depth,
                    token_idx=token_idx,
                    parent=parent,
                    draft_token=draft_token,
                    target_token=target_token,
                    use_rejection=use_rejection,
                    accept_prob=accept_prob if use_rejection else None,
                    num_accepted=num_accepted,
                    accepted_node_indices=accepted_node_indices)
                return num_accepted
            # Accept the draft token.
            new_tokens_tensor[num_accepted, seq_slot,
                              DEFAULT_BEAM_IDX] = draft_token
            request.add_new_token(draft_token, DEFAULT_BEAM_IDX)
            accepted_node_indices.append(token_idx)
            num_accepted += 1
            if self._handle_stop_criteria(request, draft_token,
                                          beam_idx=DEFAULT_BEAM_IDX,
                                          max_seq_len=self.max_seq_len):
                request.py_num_accepted_draft_tokens_indices = (
                    accepted_node_indices)
                _smc_debug(
                    "accept_selected_stop_on_draft",
                    request,
                    selected_particle=selected_particle,
                    depth=depth,
                    token_idx=token_idx,
                    draft_token=draft_token,
                    num_accepted=num_accepted,
                    accepted_node_indices=accepted_node_indices)
                return num_accepted

        request.py_num_accepted_draft_tokens_indices = accepted_node_indices

        # Whole chain accepted: emit the target's bonus token from the buffered
        # next-token at this step (the canonical accept tail, identical to
        # ``_process_draft_tokens_greedy`` and the no-rejection branch of
        # ``_process_draft_tokens_rejection_sampling``).  Indexing target_tokens
        # for the bonus is avoided: in the flat n_particles x gamma layout the
        # child of the deepest node is not at a fixed offset, so the buffered
        # next-token (which the target sampler already prepared at this step) is
        # the unambiguous source.
        new_token = add_token(request, new_tokens_list,
                              beam_idx=DEFAULT_BEAM_IDX, step=num_accepted)
        self.finish_if_reason(request, finish_reasons, step=num_accepted,
                              beam_idx=DEFAULT_BEAM_IDX)
        _smc_debug(
            "accept_selected_full_chain",
            request,
            selected_particle=selected_particle,
            bonus_token=new_token,
            num_accepted=num_accepted,
            accepted_node_indices=accepted_node_indices)
        return num_accepted

    def _compute_logprob_diff(self, request, num_accepted: int) -> torch.Tensor | None:
        if num_accepted == 0:
            return None
        draft_logits = request.py_draft_logits
        draft_token_log_probs = getattr(
            request, "py_smc_draft_token_log_probs", None)
        target_probs = request.py_target_probs
        if target_probs is None:
            return None
        if draft_logits is None and draft_token_log_probs is None:
            return None

        accepted_indices = getattr(
            request, "py_num_accepted_draft_tokens_indices", [])
        if accepted_indices:
            token_indices = [int(i) for i in accepted_indices[:num_accepted]]
        else:
            token_indices = list(range(num_accepted))

        target_probs = target_probs[token_indices].float()
        if draft_token_log_probs is not None:
            draft_token_log_probs = draft_token_log_probs[token_indices].float()
            log_prob_device = draft_token_log_probs.device
        else:
            draft_logits = draft_logits[token_indices].float()
            log_prob_device = draft_logits.device

        draft_tokens = torch.tensor(
            [int(request.py_draft_tokens[i]) for i in token_indices],
            dtype=torch.long,
            device=log_prob_device,
        )
        target_token_probs = target_probs.gather(
            1, draft_tokens.to(target_probs.device).unsqueeze(1)).squeeze(1)
        if draft_token_log_probs is None:
            draft_log_probs = torch.log_softmax(
                draft_logits / max(self.draft_temperature, 1e-6), dim=-1)
            draft_token_log_probs = draft_log_probs.gather(
                1, draft_tokens.unsqueeze(1)).squeeze(1)
        return (torch.log(target_token_probs.clamp_min(1e-30)) -
                draft_token_log_probs.to(target_probs.device)).sum()

from dataclasses import dataclass

import torch

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import (BaseResourceManager, ResourceManager,
                                           ResourceManagerType)
from ..pyexecutor.sampler import (DEFAULT_BEAM_IDX, FinishReasonsList,
                                  TorchSampler, add_token)
from .interface import SpecMetadata
from .model_drafter import ModelDrafter
from .spec_tree_manager import SpecTreeManager


def build_smc_particle_choices(gamma: int, n_particles: int) -> list[list[int]]:
    choices: list[list[int]] = []
    for depth in range(1, gamma + 1):
        for particle_idx in range(n_particles):
            choices.append([particle_idx] + [0] * (depth - 1))
    return choices

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

    def process_static_draft_outputs(self, outputs, draft_batch) -> None:
        if not isinstance(outputs, dict) or "draft_token_log_probs" not in outputs:
            super().process_static_draft_outputs(outputs, draft_batch)
            return

        draft_tokens_host = outputs["new_draft_tokens"].cpu()
        draft_token_log_probs = outputs["draft_token_log_probs"]

        for req_idx, req in enumerate(draft_batch.all_requests()):
            target_model_req = self.req_id_to_old_request[req.py_request_id]
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                continue
            target_model_req.py_draft_tokens = []
            token_log_probs = []
            for token_idx in range(self.max_total_draft_tokens):
                target_model_req.py_draft_tokens.append(
                    draft_tokens_host[token_idx][req_idx])
                token_log_probs.append(draft_token_log_probs[token_idx][req_idx])

            target_model_req.py_draft_logits = None
            target_model_req.py_smc_draft_token_log_probs = torch.stack(
                token_log_probs)


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
            for request in state.requests:
                if request.state == LlmRequestState.GENERATION_COMPLETE:
                    continue
                draft_tokens = request.py_draft_tokens
                if not draft_tokens:
                    continue
                if getattr(request, "py_target_probs", None) is None:
                    continue
                if getattr(request, "py_smc_draft_token_log_probs",
                           None) is None:
                    continue
                group_ids.append(
                    int(getattr(request, "py_smc_group_id",
                                request.py_request_id)))
                diff_rows.append(self._compute_particle_logprob_diffs(request))
            if group_ids:
                logprob_diffs = torch.stack(diff_rows, dim=0)
                selected, ess = spec_manager.select_particles_batched(
                    group_ids, logprob_diffs)
                for idx, group_id in enumerate(group_ids):
                    self._batched_selection[group_id] = (selected[idx],
                                                         ess[idx])
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
            return self._process_draft_tokens_greedy(
                request, new_tokens=new_tokens_list, finish_reasons=finish_reasons)

        if getattr(request, "py_target_probs", None) is None:
            raise RuntimeError("SMC-SD requires target token probabilities.")
        if getattr(request, "py_smc_draft_token_log_probs", None) is None:
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
        else:
            # Fallback (e.g. no SMC resource manager): per-request path.
            logprob_diffs = self._compute_particle_logprob_diffs(request)
            selected_particle = int(torch.argmax(logprob_diffs).item())
            if resource_manager is not None:
                spec_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.SPEC_RESOURCE_MANAGER)
                if isinstance(spec_manager, SMCResourceManager):
                    selected_particle, ess = spec_manager.select_particle(
                        group_id, logprob_diffs)
                    setattr(request, "py_smc_effective_sample_size", ess)
                    setattr(request, "py_smc_selected_particle",
                            selected_particle)

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
        token_indices = self._particle_token_indices[selected_particle]
        request.py_num_accepted_draft_tokens_indices = token_indices
        seq_slot = request.py_seq_slot
        assert seq_slot is not None

        num_accepted = 0
        for step, token_idx in enumerate(token_indices):
            new_token = int(request.py_draft_tokens[token_idx])
            new_tokens_tensor[step, seq_slot, DEFAULT_BEAM_IDX] = new_token
            request.add_new_token(new_token, DEFAULT_BEAM_IDX)
            num_accepted += 1
            if self._handle_stop_criteria(
                    request,
                    new_token,
                    beam_idx=DEFAULT_BEAM_IDX,
                    max_seq_len=self.max_seq_len):
                return num_accepted

        bonus_step = token_indices[-1] + 1
        new_token = add_token(request, new_tokens_list,
                              beam_idx=DEFAULT_BEAM_IDX, step=bonus_step)
        self.finish_if_reason(request, finish_reasons, step=bonus_step,
                              beam_idx=DEFAULT_BEAM_IDX)
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

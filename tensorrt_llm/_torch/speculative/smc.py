from dataclasses import dataclass

import torch

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import (BaseResourceManager, ResourceManager,
                                           ResourceManagerType)
from ..pyexecutor.sampler import FinishReasonsList, TorchSampler
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

    def reset_request(self, request_id: int) -> None:
        self.log_weights[request_id] = torch.zeros(
            (self.n_particles,), dtype=torch.float32, device="cuda")
        self.step_counts[request_id] = 0
        self.accepted_lengths[request_id] = 0

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

    def should_provide_draft_probs(self, request) -> bool:
        return True

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

        num_accepted = super().process_draft_tokens(
            request,
            new_tokens_tensor,
            new_tokens_list,
            finish_reasons,
            resource_manager,
        )

        if resource_manager is not None:
            spec_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if isinstance(spec_manager, SMCResourceManager):
                group_id = int(getattr(request, "py_smc_group_id", request.py_request_id))
                particle_idx = int(getattr(request, "py_smc_particle_idx", 0))
                spec_manager.record_acceptance(group_id, num_accepted)
                logprob_diff = self._compute_logprob_diff(request, num_accepted)
                if logprob_diff is not None:
                    spec_manager.record_logprob_diff(group_id, particle_idx, logprob_diff)
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

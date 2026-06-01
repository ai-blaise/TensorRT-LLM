from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.smc import (
    build_smc_particle_choices,
    SMCModelDrafter,
    SMCSampler,
    SMCResourceManager,
)
from tensorrt_llm._torch.speculative.utils import (
    get_spec_decoder,
    get_spec_metadata,
)
from tensorrt_llm.llmapi import SMCDecodingConfig


def _smc_config():
    return SMCDecodingConfig(
        speculative_model="BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP")


def test_smc_config_uses_gamma_draft_tokens_and_bonus_target_token():
    config = _smc_config()

    assert config.spec_dec_mode is SpeculativeDecodingMode.SMC
    assert config.max_draft_len == 6
    assert config.max_total_draft_tokens == 24
    assert config.tokens_per_gen_step == 25


def test_smc_particle_choices_are_hidden_static_tree_paths():
    choices = build_smc_particle_choices(gamma=3, n_particles=4)

    assert len(choices) == 12
    assert choices[:4] == [[0], [1], [2], [3]]
    assert choices[4:8] == [[0, 0], [1, 0], [2, 0], [3, 0]]
    assert choices[8:] == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]


def test_smc_metadata_allocates_particle_state():
    config = _smc_config()
    model_config = SimpleNamespace(vocab_size=32000, torch_dtype=torch.bfloat16)

    metadata = get_spec_metadata(
        config, model_config, max_num_requests=8, max_num_tokens=128)
    metadata.prepare()

    assert metadata.max_draft_len == 6
    assert metadata.max_total_draft_tokens == 24
    assert metadata.runtime_draft_len == 24
    assert metadata.is_spec_dec_tree
    assert not metadata.is_spec_dec_dynamic_tree
    assert tuple(metadata.particle_log_weights.shape) == (8, 4)
    assert tuple(metadata.particle_ess.shape) == (8,)


def test_smc_resource_manager_tracks_gpu_ess_and_acceptance():
    config = _smc_config()
    manager = SMCResourceManager(config, max_num_requests=4)

    manager.reset_request(17)
    assert tuple(manager.log_weights[17].shape) == (4,)
    assert float(manager.effective_sample_size(17).item()) == 4.0
    assert not bool(manager.needs_resample(17).item())

    manager.log_weights[17][0] = 20.0
    assert bool(manager.needs_resample(17).item())

    manager.record_acceptance(17, 6)
    assert manager.accepted_lengths[17] == 6
    assert manager.step_counts[17] == 1

    manager.record_logprob_diff(17, 2, torch.tensor(1.25, device="cuda"))
    assert torch.allclose(manager.log_weights[17][2], torch.tensor(1.25, device="cuda"))


def test_smc_decoder_allocates_gamma_plus_bonus_storage():
    config = _smc_config()
    args = TorchSampler.Args(
        max_seq_len=128,
        max_draft_len=config.max_draft_len,
        max_num_sequences=4,
        max_beam_width=1,
        max_total_draft_tokens=config.max_total_draft_tokens,
    )

    sampler = get_spec_decoder(args, config)

    assert isinstance(sampler, SMCSampler)
    assert sampler.max_tokens == 25
    assert sampler.gamma == 6
    assert sampler.should_provide_draft_probs(None)


def test_smc_drafter_does_not_require_tree_manager():
    class ResourceManagerStub:
        def get_resource_manager(self, _resource_type):
            return object()

    drafter = object.__new__(SMCModelDrafter)

    assert drafter.update_cur_draft_layer_idx(0, ResourceManagerStub()) is None


def test_smc_logprob_diff_uses_accepted_tree_indices():
    sampler = object.__new__(SMCSampler)
    sampler.draft_temperature = 1.0
    request = SimpleNamespace(
        py_draft_tokens=[3, 4, 5, 1],
        py_num_accepted_draft_tokens_indices=[2, 0],
        py_draft_logits=torch.full((4, 8), -20.0, device="cuda"),
        py_target_probs=torch.full((4, 8), 1e-6, device="cuda"),
    )
    request.py_draft_logits[2, 5] = 0.0
    request.py_draft_logits[0, 3] = 0.0
    request.py_target_probs[2, 5] = 0.25
    request.py_target_probs[0, 3] = 0.5

    actual = sampler._compute_logprob_diff(request, num_accepted=2)
    expected = (
        torch.log(torch.tensor(0.25, device="cuda"))
        + torch.log(torch.tensor(0.5, device="cuda"))
        - torch.log_softmax(request.py_draft_logits[2], dim=-1)[5]
        - torch.log_softmax(request.py_draft_logits[0], dim=-1)[3]
    )

    assert torch.allclose(actual, expected)


def test_smc_logprob_diff_falls_back_to_prefix_indices():
    sampler = object.__new__(SMCSampler)
    sampler.draft_temperature = 1.0
    request = SimpleNamespace(
        py_draft_tokens=[1, 2],
        py_num_accepted_draft_tokens_indices=[],
        py_draft_logits=torch.zeros((2, 4), device="cuda"),
        py_target_probs=torch.full((2, 4), 0.25, device="cuda"),
    )

    assert torch.isfinite(sampler._compute_logprob_diff(request, num_accepted=2))


def test_smc_logprob_diff_accepts_selected_token_log_probs():
    sampler = object.__new__(SMCSampler)
    sampler.draft_temperature = 1.0
    request = SimpleNamespace(
        py_draft_tokens=[7, 8, 9],
        py_num_accepted_draft_tokens_indices=[1, 2],
        py_draft_logits=None,
        py_smc_draft_token_log_probs=torch.log(
            torch.tensor([0.2, 0.25, 0.5], device="cuda")),
        py_target_probs=torch.full((3, 16), 1e-6, device="cuda"),
    )
    request.py_target_probs[1, 8] = 0.5
    request.py_target_probs[2, 9] = 0.25

    actual = sampler._compute_logprob_diff(request, num_accepted=2)
    expected = torch.log(torch.tensor(0.5 / 0.25 * 0.25 / 0.5, device="cuda"))

    assert torch.allclose(actual, expected)

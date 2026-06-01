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

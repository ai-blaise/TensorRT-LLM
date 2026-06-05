from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.sampler import (
    GREEDY,
    TorchSampler,
    _CachingRequestGrouper,
)
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


def _bare_smc_sampler(gamma=3, n_particles=2):
    sampler = object.__new__(SMCSampler)
    sampler.gamma = gamma
    sampler.n_particles = n_particles
    sampler.resample_threshold = 0.5
    sampler.draft_temperature = 1.0
    sampler.max_seq_len = 128
    # The rejection-sampling accept path (production default) draws u ~ U(0,1)
    # via the base TorchSampler RNG. object.__new__ skips __init__, so seed the
    # same lazily-initialized generator state the real __init__ sets up.
    sampler._generator = None
    sampler._global_seed = 0
    sampler._particle_token_indices = [[depth * n_particles + particle
                                        for depth in range(gamma)]
                                       for particle in range(n_particles)]
    return sampler


def test_smc_particle_logprob_diffs_score_full_paths():
    sampler = _bare_smc_sampler(gamma=3, n_particles=2)
    request = SimpleNamespace(
        py_draft_tokens=[1, 2, 3, 4, 5, 6],
        py_smc_draft_token_log_probs=torch.log(
            torch.full((6,), 0.5, device="cuda")),
        py_target_probs=torch.full((7, 8), 1e-4, device="cuda"),
    )
    request.py_target_probs[0, 2] = 0.8
    request.py_target_probs[2, 4] = 0.7
    request.py_target_probs[4, 6] = 0.6

    diffs = sampler._compute_particle_logprob_diffs(request)

    assert diffs.shape == (2,)
    assert diffs[1] > diffs[0]


def test_smc_sampler_advances_selected_particle_without_prefix_rejection():
    # Production default (rejection sampling): the selected particle's first two
    # draft tokens have q/p >= 1 (target prob 0.9/0.8 vs draft 0.5) so they are
    # accepted with prob 1.0 -> deterministic accept of exactly 2, regardless of
    # the random draw; the third has q=1e-4 (reject). _bare_smc_sampler seeds the
    # base-sampler RNG so the rejection path is exercisable here.
    sampler = _bare_smc_sampler(gamma=2, n_particles=2)
    sampler.finish_if_reason = lambda *args, **kwargs: False
    sampler._handle_stop_criteria = lambda *args, **kwargs: False

    class Request:
        py_request_id = 11
        py_seq_slot = 0
        py_draft_tokens = [10, 20, 11, 21]
        py_smc_draft_token_log_probs = torch.log(
            torch.full((4,), 0.5, device="cuda"))
        py_target_probs = torch.full((5, 32), 1e-4, device="cuda")

        def __init__(self):
            self.tokens = []
            self.py_num_accepted_draft_tokens_indices = []

        def add_new_token(self, token, _beam_idx):
            self.tokens.append(int(token))

    request = Request()
    request.py_target_probs[0, 20] = 0.9
    request.py_target_probs[2, 21] = 0.8
    new_tokens_tensor = torch.zeros((5, 1, 1), dtype=torch.int32, device="cuda")
    new_tokens_list = [[[0]] for _ in range(5)]
    # After accepting 2 draft tokens the bonus target token is read from
    # new_tokens_list[num_accepted] == new_tokens_list[2] (see add_token).
    new_tokens_list[2][0][0] = 99

    accepted = sampler.process_draft_tokens(
        request,
        new_tokens_tensor,
        new_tokens_list,
        finish_reasons=new_tokens_list,
    )

    assert accepted == 2
    assert request.py_num_accepted_draft_tokens_indices == [1, 3]
    assert request.tokens == [20, 21, 99]


def test_smc_sampler_rejection_path_accepts_and_is_finite(monkeypatch):
    # Production default: rejection sampling accepts draft token t at depth d
    # with prob min(1, q/p), q = P_target(t | parent), p = P_draft(t). Validate
    # the mechanism numerically: when q/p >= 1 every selected-chain token is
    # accepted (the bonus target token follows), emitted tokens are finite and
    # in-vocab, and accepted_length is bounded by gamma.
    monkeypatch.setenv("SMC_REJECTION_ACCEPT", "1")
    gamma, n_particles, vocab = 6, 4, 32
    sampler = _bare_smc_sampler(gamma=gamma, n_particles=n_particles)
    sampler.finish_if_reason = lambda *args, **kwargs: False
    sampler._handle_stop_criteria = lambda *args, **kwargs: False

    num_nodes = gamma * n_particles + 1
    # Selected particle is index 0; its chain tokens are draft positions
    # [0, n_particles, 2*n_particles, ...]. Make the draft cheap (p small) and
    # the target love those exact tokens (q large) so q/p >= 1 -> always accept.
    draft_tokens = [0] * (gamma * n_particles)
    sel_positions = [d * n_particles for d in range(gamma)]
    for d, pos in enumerate(sel_positions):
        draft_tokens[pos] = 5 + d  # in-vocab, distinct per depth

    class Request:
        py_request_id = 7
        py_seq_slot = 0

        def __init__(self):
            self.py_draft_tokens = list(draft_tokens)
            self.py_smc_draft_token_log_probs = torch.log(
                torch.full((gamma * n_particles,), 0.1, device="cuda"))
            self.py_target_probs = torch.full((num_nodes, vocab),
                                              1e-4,
                                              device="cuda")
            self.tokens = []
            self.py_num_accepted_draft_tokens_indices = []

        def add_new_token(self, token, _beam_idx):
            self.tokens.append(int(token))

    request = Request()
    # parent_steps for the selected chain: depth 0 -> node 0, depth d -> the
    # previous chain node + 1. Mirror _get_particle_index_tensors so the target
    # strongly prefers each draft token at its parent node.
    _, parent_steps = sampler._get_particle_index_tensors(
        request.py_target_probs.device)
    parents_sel = parent_steps[0].tolist()
    for d, pos in enumerate(sel_positions):
        request.py_target_probs[parents_sel[d], draft_tokens[pos]] = 0.95

    new_tokens_tensor = torch.zeros((num_nodes, 1, 1),
                                    dtype=torch.int32,
                                    device="cuda")
    new_tokens_list = [[[0]] for _ in range(num_nodes)]
    bonus_token = 13  # in-vocab bonus token slot
    new_tokens_list[gamma][0][0] = bonus_token

    accepted = sampler.process_draft_tokens(
        request,
        new_tokens_tensor,
        new_tokens_list,
        finish_reasons=new_tokens_list,
    )

    # q/p = 0.95 / 0.1 -> accept_prob clamped to 1.0 at every depth.
    assert accepted == gamma
    assert 0 <= accepted <= gamma
    assert request.tokens[:gamma] == [draft_tokens[p] for p in sel_positions]
    assert request.tokens[gamma] == bonus_token  # bonus target token
    assert all(0 <= t < vocab for t in request.tokens)  # in-vocab
    assert all(t == t for t in request.tokens)  # finite (no NaN)


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


def test_smc_greedy_requests_still_request_target_probabilities():
    grouper = _CachingRequestGrouper(max_num_sequences=1)
    store = grouper._store
    store.strategies[0] = GREEDY
    request = SimpleNamespace(
        py_draft_tokens=[1, 2, 3],
        py_smc_draft_token_log_probs=torch.zeros((3,), device="cuda"),
    )

    groups = grouper.group_requests_by_strategy_key(
        [request],
        strategy_to_key=lambda strategy: strategy,
        pin_memory=False,
        seq_slots=torch.tensor([0], dtype=torch.int64),
        vocab_size=32000,
    )

    assert len(groups) == 1
    group_key, group_value = next(iter(groups.items()))
    assert group_key.needs_probs
    assert group_value.speculation_needs_probs_indices.tolist() == [0]


def test_smc_sampler_disables_fast_greedy_path():
    sampler = object.__new__(SMCSampler)

    assert not sampler._can_use_fast_greedy_path([SimpleNamespace()])

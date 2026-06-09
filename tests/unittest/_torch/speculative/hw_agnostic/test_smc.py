import inspect
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.sampler import (
    GREEDY,
    TorchSampler,
    _CachingRequestGrouper,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.drafting_loops import (
    _mark_attn_metadata_generation_only,
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
from tensorrt_llm._torch.pyexecutor import py_executor_creator
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


def test_smc_config_ports_sglang_draft_kv_dtype_aliases():
    bf16 = SMCDecodingConfig(
        speculative_model="BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP",
        draft_kv_cache_dtype="bf16")
    higgs = SMCDecodingConfig(
        speculative_model="BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP",
        draft_kv_cache_dtype="higgs_2bit")

    assert bf16.draft_kv_cache_dtype == "bfloat16"
    assert higgs.draft_kv_cache_dtype == "kvarn_k2v2_g128"


def test_smc_config_accepts_explicit_optrt_gqa_kvarn_dtype():
    config = SMCDecodingConfig(
        speculative_model="BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP",
        draft_kv_cache_dtype="kvarn_k2v2_g128")

    assert config.draft_kv_cache_dtype == "kvarn_k2v2_g128"


def test_smc_mode_admits_overlap_scheduler():
    assert SpeculativeDecodingMode.SMC.support_overlap_scheduler()


def test_smc_creator_does_not_force_disable_overlap_scheduler():
    source = inspect.getsource(py_executor_creator.create_py_executor)

    assert "Disabling overlap scheduler for SMC-SD" not in source


def test_smc_creator_preserves_gqa_kvarn_draft_kv_dispatch():
    source = inspect.getsource(py_executor_creator.create_py_executor)

    assert 'startswith("kvarn_")' in source
    assert "draft_llm_args.kv_cache_config.tokens_per_block = 128" in source


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


def test_smc_resource_manager_keeps_zombie_until_complete():
    config = _smc_config()
    manager = SMCResourceManager(config, max_num_requests=4)
    manager.reset_request(17)

    request = SimpleNamespace(
        py_request_id=17,
        state=LlmRequestState.GENERATION_TO_COMPLETE,
    )
    batch = SimpleNamespace(all_requests=lambda: [request])

    manager.update_resources(batch)
    assert 17 in manager.log_weights

    request.state = LlmRequestState.GENERATION_COMPLETE
    manager.update_resources(batch)
    assert 17 not in manager.log_weights


def test_smc_overlap_static_draft_commit_uses_evented_host_tokens():
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 2

    class Event:
        def __init__(self):
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

    event = Event()
    host_tokens = torch.tensor([[11], [12]], dtype=torch.int64)
    sample_state = SimpleNamespace(
        host=SimpleNamespace(new_tokens=host_tokens),
        sampler_event=event,
    )
    outputs = {
        "new_draft_tokens": torch.tensor([[99], [98]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1], [-0.2]]),
        "sample_state": sample_state,
    }
    target_request = SimpleNamespace(
        py_request_id=7,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_tokens=[],
    )
    drafter.req_id_to_old_request = {7: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=7)])

    drafter.process_static_draft_outputs(outputs, draft_batch)

    assert event.synchronized
    assert [int(token) for token in target_request.py_draft_tokens] == [11, 12]
    assert torch.allclose(target_request.py_smc_draft_token_log_probs,
                          torch.tensor([-0.1, -0.2]))
    assert target_request.py_draft_logits is None


def test_smc_overlap_pack_does_not_require_generic_draft_logits():
    drafter = object.__new__(SMCModelDrafter)
    sample_state = SimpleNamespace(
        host=SimpleNamespace(new_tokens=torch.tensor([[11]], dtype=torch.int64)),
        sampler_event=SimpleNamespace(synchronize=lambda: None),
    )
    log_probs = torch.tensor([[-0.1]])
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": log_probs,
    }

    packed = drafter._pack_static_draft_outputs_for_overlap(
        outputs, sample_state)

    assert packed["sample_state"] is sample_state
    assert packed["draft_token_log_probs"] is log_probs
    assert "draft_logits" not in packed


def test_smc_direct_static_draft_commit_packs_evented_host_tokens():
    drafter = object.__new__(SMCModelDrafter)
    drafter.draft_model_engine = object()
    drafter.use_static_draft_loop = True
    draft_request = SimpleNamespace(py_request_id=7)
    draft_batch = SimpleNamespace(all_requests=lambda: [draft_request])
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
    }
    sample_state = SimpleNamespace(
        host=SimpleNamespace(new_tokens=torch.tensor([[11]], dtype=torch.int64)),
        sampler_event=SimpleNamespace(synchronize=lambda: None),
    )
    captured = {}
    freed = []

    drafter._setup_draft_batch_and_resources = lambda _batch: draft_batch
    drafter.update_cur_draft_layer_idx = lambda *_args, **_kwargs: None
    drafter.forward_draft_model = lambda *_args, **_kwargs: outputs
    drafter._create_static_draft_sample_state = (
        lambda actual_outputs, actual_batch: sample_state)
    drafter.process_static_draft_outputs = (
        lambda actual_outputs, actual_batch: captured.update(
            outputs=actual_outputs, draft_batch=actual_batch))
    drafter.draft_seq_slot_manager = SimpleNamespace(
        free_resources=lambda req: freed.append(req))

    drafter.prepare_draft_tokens(SimpleNamespace(), object())

    assert captured["draft_batch"] is draft_batch
    assert captured["outputs"]["sample_state"] is sample_state
    assert captured["outputs"]["draft_token_log_probs"] is outputs[
        "draft_token_log_probs"]
    assert "draft_logits" not in captured["outputs"]
    assert freed == [draft_request]


def test_smc_overlap_static_draft_commit_skips_prefill_context():
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 1

    sample_state = SimpleNamespace(
        host=SimpleNamespace(new_tokens=torch.tensor([[11]], dtype=torch.int64)),
        sampler_event=SimpleNamespace(synchronize=lambda: None),
    )
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
        "sample_state": sample_state,
    }
    target_request = SimpleNamespace(
        py_request_id=9,
        state=LlmRequestState.CONTEXT_INIT,
        py_draft_tokens=["unchanged"],
    )
    drafter.req_id_to_old_request = {9: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=9)])

    drafter.process_static_draft_outputs(outputs, draft_batch)

    assert target_request.py_draft_tokens == ["unchanged"]


def test_smc_generation_only_metadata_moves_context_blocks_to_decode():
    metadata = SimpleNamespace(
        num_contexts=1,
        host_request_types=torch.zeros(1, dtype=torch.int32),
        num_context_blocks=32,
        num_generation_blocks=0,
    )

    _mark_attn_metadata_generation_only(metadata)

    assert metadata.num_contexts == 0
    assert metadata.host_request_types.tolist() == [1]
    assert metadata.num_context_blocks == 0
    assert metadata.num_generation_blocks == 32


def test_smc_model_drafter_skips_attention_dp_dummy_padding():
    drafter = object.__new__(SMCModelDrafter)
    dummy_request = SimpleNamespace(
        py_request_id=0,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_disable_speculative_decoding=False,
        is_attention_dp_dummy=True,
        py_last_draft_tokens=[1] * 24,
        py_draft_pages_allocated=24,
    )
    scheduled_batch = SimpleNamespace(all_requests=lambda: [dummy_request])

    def fail_if_called(_request):
        raise AssertionError("attention-DP dummy entered draft request setup")

    drafter._create_draft_request_for_request = fail_if_called

    draft_batch = drafter._prepare_draft_batch(scheduled_batch)

    assert draft_batch.batch_size == 0


def test_smc_overlap_static_draft_commit_skips_aborted_or_zombie_request():
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 1
    event = SimpleNamespace(synchronize=lambda: None)
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
        "sample_state": SimpleNamespace(
            host=SimpleNamespace(new_tokens=torch.tensor([[11]],
                                                        dtype=torch.int64)),
            sampler_event=event,
        ),
    }
    target_request = SimpleNamespace(
        py_request_id=10,
        state=LlmRequestState.GENERATION_COMPLETE,
        py_draft_tokens=["old"],
        py_draft_logits="old_logits",
        py_smc_draft_token_log_probs="old_log_probs",
    )
    drafter.req_id_to_old_request = {10: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=10)])

    drafter.process_static_draft_outputs(outputs, draft_batch)

    assert target_request.py_draft_tokens == ["old"]
    assert target_request.py_draft_logits == "old_logits"
    assert target_request.py_smc_draft_token_log_probs == "old_log_probs"


def test_smc_overlap_commit_preserves_disagg_pin_and_kvarn_metadata():
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 1
    event = SimpleNamespace(synchronize=lambda: None)
    pin_metadata = SimpleNamespace(
        disagg_request_id="ctx-42",
        prefill_worker="prefill-a",
        decode_worker="decode-b",
        remote_block_ids=(3, 5, 8),
    )
    kvarn_metadata = {
        "mla_latent_kv_dtype": "kvarn_k2v2",
        "mla_latent_kv_amortize": True,
    }
    target_request = SimpleNamespace(
        py_request_id=11,
        py_smc_group_id=211,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_tokens=[],
        py_disaggregated_params=pin_metadata,
        py_kvarn_metadata=kvarn_metadata,
    )
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
        "sample_state": SimpleNamespace(
            host=SimpleNamespace(new_tokens=torch.tensor([[11]],
                                                        dtype=torch.int64)),
            sampler_event=event,
        ),
    }
    drafter.req_id_to_old_request = {11: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=11)])

    drafter.process_static_draft_outputs(outputs, draft_batch)

    assert target_request.py_disaggregated_params is pin_metadata
    assert target_request.py_kvarn_metadata is kvarn_metadata
    assert target_request.py_smc_group_id == 211
    assert target_request.py_draft_logits is None
    assert [int(token) for token in target_request.py_draft_tokens] == [11]
    assert torch.allclose(target_request.py_smc_draft_token_log_probs,
                          torch.tensor([-0.1]))


def test_smc_overlap_commit_requires_generation_pin_before_mutation(monkeypatch):
    monkeypatch.delenv("TRTLLM_SMC_REQUIRE_REQUEST_PIN", raising=False)
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 1
    target_request = SimpleNamespace(
        py_request_id=12,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_tokens=["old"],
        py_draft_logits="old_logits",
        py_smc_draft_token_log_probs="old_log_probs",
        py_disaggregated_params=SimpleNamespace(
            request_type="generation_only",
            disagg_request_id="ctx-12",
            ctx_dp_rank=None,
            ctx_info_endpoint="nixl://ctx/12",
        ),
        is_generation_only_request=lambda: True,
    )
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
        "sample_state": SimpleNamespace(
            host=SimpleNamespace(new_tokens=torch.tensor([[11]], dtype=torch.int64)),
            sampler_event=SimpleNamespace(synchronize=lambda: None),
        ),
    }
    drafter.req_id_to_old_request = {12: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=12)])

    try:
        drafter.process_static_draft_outputs(outputs, draft_batch)
    except RuntimeError as exc:
        assert "missing ctx_dp_rank" in str(exc)
    else:
        raise AssertionError("missing ctx_dp_rank did not fail closed")

    assert target_request.py_draft_tokens == ["old"]
    assert target_request.py_draft_logits == "old_logits"
    assert target_request.py_smc_draft_token_log_probs == "old_log_probs"


def test_smc_overlap_commit_preserves_valid_generation_pin_metadata(monkeypatch):
    monkeypatch.delenv("TRTLLM_SMC_REQUIRE_REQUEST_PIN", raising=False)
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 1
    pin_metadata = SimpleNamespace(
        request_type="generation_only",
        disagg_request_id="ctx-13",
        ctx_request_id="ctx-13",
        ctx_dp_rank=2,
        ctx_info_endpoint=("nixl://ctx/13",),
    )
    target_request = SimpleNamespace(
        py_request_id=13,
        py_smc_group_id=213,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_tokens=[],
        py_disaggregated_params=pin_metadata,
        py_kvarn_metadata={"mla_latent_kv_dtype": "kvarn_k2v2"},
        is_generation_only_request=lambda: True,
    )
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
        "sample_state": SimpleNamespace(
            host=SimpleNamespace(new_tokens=torch.tensor([[11]], dtype=torch.int64)),
            sampler_event=SimpleNamespace(synchronize=lambda: None),
        ),
    }
    drafter.req_id_to_old_request = {13: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=13)])

    drafter.process_static_draft_outputs(outputs, draft_batch)

    assert target_request.py_disaggregated_params is pin_metadata
    assert target_request.py_smc_group_id == 213
    assert [int(token) for token in target_request.py_draft_tokens] == [11]
    assert torch.allclose(target_request.py_smc_draft_token_log_probs,
                          torch.tensor([-0.1]))


def test_smc_overlap_commit_requires_evented_pinned_sample_state(monkeypatch):
    monkeypatch.delenv("TRTLLM_SMC_ALLOW_UNPINNED_DRAFT_COMMIT", raising=False)
    drafter = object.__new__(SMCModelDrafter)
    drafter.max_total_draft_tokens = 1
    target_request = SimpleNamespace(
        py_request_id=14,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_tokens=["old"],
        py_draft_logits="old_logits",
        py_smc_draft_token_log_probs="old_log_probs",
    )
    outputs = {
        "new_draft_tokens": torch.tensor([[99]], dtype=torch.int64),
        "draft_token_log_probs": torch.tensor([[-0.1]]),
    }
    drafter.req_id_to_old_request = {14: target_request}
    draft_batch = SimpleNamespace(
        all_requests=lambda: [SimpleNamespace(py_request_id=14)])

    try:
        drafter.process_static_draft_outputs(outputs, draft_batch)
    except RuntimeError as exc:
        assert "requires evented pinned host" in str(exc)
    else:
        raise AssertionError("missing pinned sample_state did not fail closed")

    assert target_request.py_draft_tokens == ["old"]
    assert target_request.py_draft_logits == "old_logits"
    assert target_request.py_smc_draft_token_log_probs == "old_log_probs"


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

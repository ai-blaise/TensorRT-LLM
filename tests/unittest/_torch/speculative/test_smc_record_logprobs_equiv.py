"""Element-for-element equivalence test for the optimized
``SMCStaticParticleDraftingLoopWrapper._record_layer_draft_log_probs``.

What changed in the method under test:
  1. Sync elimination (DEFAULT, bit-identical): the per-call O(cur_draft_idx)
     re-summation of ``top_k_list[*].tolist()`` (one device->host copy per
     layer, i.e. O(gamma^2) syncs per decode step) is replaced by a host-side
     prefix sum cached once on the manager, mirroring the existing device
     tensor ``draft_tokens_indices_cumsum``. The arithmetic is untouched.
  2. Vectorized logsumexp+gather (OPT-IN via
     ``spec_tree_manager.smc_vectorize_logprob_record = True``): for a uniform
     layer (each parent exactly one child) the per-parent Python loop collapses
     to a single batched logsumexp + gather. The gather is bit-exact; the
     batched logsumexp reduces a contiguous tensor instead of per-parent
     non-contiguous slices, which PyTorch may tile differently, so the
     normalizer can differ by up to 1 ULP (~4.77e-7). Hence the opt-in path is
     gated with atol=1e-6 (and the gather component is checked bit-exact).

The reference is a frozen copy of the ORIGINAL method, run against randomized
SMC spec-tree state across several gamma/n_particles/batch combinations.
"""
import sys

import pytest
import torch

from tensorrt_llm._torch.speculative.drafting_loops import (
    SMCStaticParticleDraftingLoopWrapper)
from tensorrt_llm._torch.speculative.smc import build_smc_particle_choices
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager


# ---------------------------------------------------------------------------
# Frozen copy of the ORIGINAL implementation (pre-optimization), the golden
# reference. Byte-faithful to the logic that shipped before this change.
# ---------------------------------------------------------------------------
def reference_record_layer_draft_log_probs(self, draft_log_prob_buffer, logits,
                                           batch_size, cur_draft_idx,
                                           spec_tree_manager):
    start = 0
    for layer_idx in range(cur_draft_idx):
        start += sum(int(repeat) for repeat in
                     spec_tree_manager.top_k_list[layer_idx].tolist())
    repeats = [int(repeat) for repeat in
               spec_tree_manager.top_k_list[cur_draft_idx].tolist()]
    end = start + sum(repeats)
    if start == end:
        return

    if cur_draft_idx == 0:
        parent_logits = logits.reshape(batch_size,
                                       logits.shape[-1]).unsqueeze(1)
    else:
        parent_indices = spec_tree_manager.tokens_gather_idx_for_drafter_model[
            cur_draft_idx].long()
        parent_logits = logits.reshape(
            batch_size, self.max_total_draft_tokens + 1,
            logits.shape[-1]).index_select(1, parent_indices)

    child_tokens = self.draft_tokens_buffer[:batch_size, start:end].long()
    cursor = 0
    for parent_idx, repeat in enumerate(repeats):
        parent = parent_logits[:, parent_idx, :].float()
        normalizer = torch.logsumexp(parent, dim=-1, keepdim=True)
        next_cursor = cursor + int(repeat)
        selected = parent.gather(1, child_tokens[:, cursor:next_cursor])
        draft_log_prob_buffer[:, start + cursor:start + next_cursor] = (
            selected - normalizer)
        cursor = next_cursor


class _Harness:
    """Minimal stand-in for the wrapper, carrying only what
    ``_record_layer_draft_log_probs`` reads off ``self``."""

    def __init__(self, max_total_draft_tokens, max_draft_len,
                 draft_tokens_buffer):
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len
        self.draft_tokens_buffer = draft_tokens_buffer

    # Class-level default for the opt-in vectorization flag, mirroring the
    # method under test.
    smc_vectorize_logprob_record = (
        SMCStaticParticleDraftingLoopWrapper.smc_vectorize_logprob_record)

    _host_draft_layout = staticmethod(
        SMCStaticParticleDraftingLoopWrapper._host_draft_layout)
    _record_layer_draft_log_probs = (
        SMCStaticParticleDraftingLoopWrapper._record_layer_draft_log_probs)


def _build_manager(gamma, n_particles, vectorize):
    choices = build_smc_particle_choices(gamma, n_particles)
    mgr = SpecTreeManager(
        max_num_requests=8,
        use_dynamic_tree=False,
        max_total_draft_tokens=len(choices),
        max_draft_len=gamma,
        eagle_choices=choices,
        dynamic_tree_max_topK=0,
    )
    mgr.smc_vectorize_logprob_record = vectorize
    return mgr


def _run_case(gamma, n_particles, batch_size, vocab, seed, vectorize, device):
    torch.manual_seed(seed)
    mgr_new = _build_manager(gamma, n_particles, vectorize)
    mgr_ref = _build_manager(gamma, n_particles, False)
    max_total = mgr_new.max_total_draft_tokens

    draft_tokens_buffer = torch.randint(
        0, vocab, (batch_size, max_total + 1), dtype=torch.long, device=device)

    harness_new = _Harness(max_total, gamma, draft_tokens_buffer)
    harness_ref = _Harness(max_total, gamma, draft_tokens_buffer)

    buf_new = torch.full((batch_size, max_total), float("nan"),
                         dtype=torch.float32, device=device)
    buf_ref = torch.full((batch_size, max_total), float("nan"),
                         dtype=torch.float32, device=device)

    for cur_draft_idx in range(gamma):
        # Match the shapes production feeds in:
        #   layer 0  -> [batch_size, vocab]
        #   layer >0 -> [batch_size * (max_total + 1), vocab]
        if cur_draft_idx == 0:
            logits = torch.randn(batch_size, vocab,
                                 dtype=torch.float32, device=device)
        else:
            logits = torch.randn(batch_size * (max_total + 1), vocab,
                                 dtype=torch.float32, device=device)
        # bf16 round-trip to mimic real draft logits; both paths see the
        # identical input, so this only stresses the value distribution.
        logits = logits.to(torch.bfloat16).to(torch.float32)

        reference_record_layer_draft_log_probs(
            harness_ref, buf_ref, logits, batch_size, cur_draft_idx, mgr_ref)
        harness_new._record_layer_draft_log_probs(
            buf_new, logits, batch_size, cur_draft_idx, mgr_new)

    torch.cuda.synchronize()
    return buf_ref, buf_new, mgr_new


def _has_uniform_layer(mgr):
    offsets, repeats_per_layer = mgr._record_logprob_host_layout
    for layer in range(1, len(repeats_per_layer)):
        reps = repeats_per_layer[layer]
        width = offsets[layer + 1] - offsets[layer]
        if reps and all(r == 1 for r in reps) and len(reps) == width:
            return True
    return False


CASES = [
    (6, 4),
    (6, 8),
    (4, 4),
    (8, 2),
    (5, 3),
]


@pytest.mark.parametrize("gamma,n_particles", CASES)
def test_record_logprobs_default_bit_identical(gamma, n_particles):
    """DEFAULT path (sync-free offsets, per-parent math): bit-for-bit equal."""
    device = "cuda"
    for seed in range(4):
        for batch_size in (1, 3):
            buf_ref, buf_new, mgr = _run_case(
                gamma, n_particles, batch_size, vocab=257, seed=seed,
                vectorize=False, device=device)
            max_abs_diff = (buf_ref - buf_new).abs().max().item()
            assert torch.equal(buf_ref, buf_new), (
                "DEFAULT path not bit-identical: gamma=%d n_particles=%d "
                "batch=%d seed=%d max_abs_diff=%r"
                % (gamma, n_particles, batch_size, seed, max_abs_diff))


@pytest.mark.parametrize("gamma,n_particles", CASES)
def test_record_logprobs_vectorized_within_1ulp(gamma, n_particles):
    """OPT-IN vectorized path: matches reference to <=1 ULP (atol=1e-6)."""
    device = "cuda"
    exercised = False
    worst = 0.0
    for seed in range(4):
        for batch_size in (1, 3):
            buf_ref, buf_new, mgr = _run_case(
                gamma, n_particles, batch_size, vocab=257, seed=seed,
                vectorize=True, device=device)
            worst = max(worst, (buf_ref - buf_new).abs().max().item())
            assert torch.allclose(buf_ref, buf_new, rtol=0.0, atol=1e-6), (
                "vectorized path exceeded 1e-6: gamma=%d n_particles=%d "
                "batch=%d seed=%d max_abs_diff=%r"
                % (gamma, n_particles, batch_size, seed,
                   (buf_ref - buf_new).abs().max().item()))
            if _has_uniform_layer(mgr):
                exercised = True
    assert exercised, "vectorized fast path never exercised; test is vacuous"
    # Document the observed worst-case deviation for this config.
    print("gamma=%d n_particles=%d vectorized worst |diff|=%.3e"
          % (gamma, n_particles, worst))


def test_vectorized_gather_component_is_bit_exact():
    """The gather half of the vectorized path (selected log-prob, before the
    normalizer) must be exactly equal to the per-parent gather. Isolating it
    confirms the only source of the <=1 ULP delta is the batched logsumexp."""
    device = "cuda"
    torch.manual_seed(7)
    gamma, n_particles = 6, 4
    mgr = _build_manager(gamma, n_particles, True)
    max_total = mgr.max_total_draft_tokens
    batch_size, vocab = 3, 257
    offsets, repeats_per_layer = (
        SMCStaticParticleDraftingLoopWrapper._host_draft_layout(mgr))
    draft_tokens_buffer = torch.randint(
        0, vocab, (batch_size, max_total + 1), dtype=torch.long, device=device)
    for cur in range(1, gamma):
        reps = repeats_per_layer[cur]
        width = offsets[cur + 1] - offsets[cur]
        if not (reps and all(r == 1 for r in reps) and len(reps) == width):
            continue
        start, end = offsets[cur], offsets[cur + 1]
        logits = torch.randn(batch_size * (max_total + 1), vocab,
                             device=device).to(torch.bfloat16).to(torch.float32)
        parent_indices = mgr.tokens_gather_idx_for_drafter_model[cur].long()
        parent_logits = logits.reshape(
            batch_size, max_total + 1, vocab).index_select(1, parent_indices)
        child_tokens = draft_tokens_buffer[:batch_size, start:end].long()
        # batched gather
        sel_vec = parent_logits.float().gather(
            -1, child_tokens.unsqueeze(-1)).squeeze(-1)
        # per-parent gather
        sel_ref = torch.empty(batch_size, end - start, device=device)
        cursor = 0
        for p, r in enumerate(reps):
            nc = cursor + int(r)
            sel_ref[:, cursor:nc] = parent_logits[:, p, :].float().gather(
                1, child_tokens[:, cursor:nc])
            cursor = nc
        assert torch.equal(sel_vec, sel_ref), (
            "batched gather diverged from per-parent gather at layer %d" % cur)


def test_fallback_path_nonuniform():
    """A hand-built non-uniform tree (a parent with 2 children) routes through
    the general per-parent path even with vectorization on, and stays
    bit-identical to the reference."""
    device = "cuda"
    torch.manual_seed(123)
    choices = [[0], [1], [0, 0], [0, 1]]
    mgr = SpecTreeManager(
        max_num_requests=4, use_dynamic_tree=False,
        max_total_draft_tokens=len(choices), max_draft_len=2,
        eagle_choices=choices, dynamic_tree_max_topK=0)
    mgr.smc_vectorize_logprob_record = True  # on, but layer is non-uniform
    max_total = mgr.max_total_draft_tokens
    batch_size, vocab = 2, 64
    draft_tokens_buffer = torch.randint(
        0, vocab, (batch_size, max_total + 1), dtype=torch.long, device=device)
    hn = _Harness(max_total, 2, draft_tokens_buffer)
    hr = _Harness(max_total, 2, draft_tokens_buffer)
    bn = torch.full((batch_size, max_total), float("nan"),
                    dtype=torch.float32, device=device)
    br = torch.full((batch_size, max_total), float("nan"),
                    dtype=torch.float32, device=device)
    for cur in range(2):
        if cur == 0:
            logits = torch.randn(batch_size, vocab, device=device)
        else:
            logits = torch.randn(batch_size * (max_total + 1), vocab,
                                 device=device)
        reference_record_layer_draft_log_probs(hr, br, logits, batch_size, cur,
                                               mgr)
        hn._record_layer_draft_log_probs(bn, logits, batch_size, cur, mgr)
    torch.cuda.synchronize()
    _, repeats_per_layer = mgr._record_logprob_host_layout
    # layer 1 top_k == [2]: one parent, two children -> non-uniform -> fallback.
    assert repeats_per_layer[1] == [2], repeats_per_layer
    assert torch.equal(br, bn)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

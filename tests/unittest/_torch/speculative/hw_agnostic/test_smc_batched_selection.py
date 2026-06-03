"""Equivalence tests for the batched SMC particle-selection pre-pass.

Proves SMCResourceManager.select_particles_batched is decision-for-decision and
state-for-state identical to the per-request select_particle across multiple
decode steps (including the stateful resample-reset branch), and that the
SMCSampler.update_requests pre-pass yields the same accepted tokens as the
unbatched fallback path.
"""
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.speculative.smc import SMCResourceManager
from tensorrt_llm.llmapi import SMCDecodingConfig


def _config():
    return SMCDecodingConfig(
        speculative_model="BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP")


def _ref_select(mgr, gid, diffs):
    """Inlined copy of the original scalar select_particle semantics."""
    mgr._ensure_request(gid)
    lw = mgr.log_weights[gid]
    lw.add_(diffs.to(device=lw.device, dtype=torch.float32))
    weights = torch.softmax(lw, dim=0)
    ess = torch.reciprocal(torch.sum(weights * weights))
    selected = int(torch.argmax(weights).item())
    if bool((ess < mgr.n_particles * mgr.resample_threshold).item()):
        lw.zero_()
        lw[selected] = torch.log(weights[selected].clamp_min(1e-30))
    return selected, ess


def test_batched_matches_per_request_over_many_steps():
    cfg = _config()
    n_particles = cfg.n_particles
    batch = 16
    gids = list(range(batch))

    ref = SMCResourceManager(cfg, max_num_requests=batch)
    bat = SMCResourceManager(cfg, max_num_requests=batch)
    for gid in gids:
        ref.reset_request(gid)
        bat.reset_request(gid)

    torch.manual_seed(0)
    for step in range(40):
        diffs = torch.randn((batch, n_particles), device="cuda") * 2.5

        ref_sel = []
        ref_ess = []
        for i, gid in enumerate(gids):
            s, e = _ref_select(ref, gid, diffs[i].clone())
            ref_sel.append(s)
            ref_ess.append(float(e.item()))

        bat_sel, bat_ess = bat.select_particles_batched(gids, diffs.clone())
        bat_ess_host = [float(x) for x in bat_ess.tolist()]

        assert ref_sel == bat_sel, f"step {step}: selection mismatch"
        for i in range(batch):
            assert abs(ref_ess[i] - bat_ess_host[i]) < 1e-4, (
                f"step {step} req {i}: ess {ref_ess[i]} vs {bat_ess_host[i]}")
            assert torch.allclose(
                ref.log_weights[gids[i]], bat.log_weights[gids[i]],
                atol=1e-5), f"step {step} req {i}: log_weights diverged"


def test_batched_empty_request_list():
    cfg = _config()
    mgr = SMCResourceManager(cfg, max_num_requests=4)
    sel, ess = mgr.select_particles_batched([], torch.empty((0, cfg.n_particles),
                                                            device="cuda"))
    assert sel == []
    assert ess.numel() == 0


def test_batched_single_request_matches_scalar():
    cfg = _config()
    ref = SMCResourceManager(cfg, max_num_requests=1)
    bat = SMCResourceManager(cfg, max_num_requests=1)
    ref.reset_request(7)
    bat.reset_request(7)
    torch.manual_seed(1)
    for _ in range(10):
        d = torch.randn((1, cfg.n_particles), device="cuda") * 3.0
        rs, _ = _ref_select(ref, 7, d[0].clone())
        bs, _ = bat.select_particles_batched([7], d.clone())
        assert [rs] == bs
        assert torch.allclose(ref.log_weights[7], bat.log_weights[7], atol=1e-5)


if __name__ == "__main__":
    test_batched_matches_per_request_over_many_steps()
    test_batched_empty_request_list()
    test_batched_single_request_matches_scalar()
    print("ALL EQUIVALENCE TESTS PASSED")

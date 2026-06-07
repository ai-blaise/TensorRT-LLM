#!/usr/bin/env python3
"""Static SMC-SD/Moondream decode pinning audit for r20.

This no-traffic audit checks that the SMC decode overlap path is wired to the
same request pin metadata required by the non-MORI NIXL completed-prefill gate.
It is intentionally source-based so it can run before a canary is ready, but it
does not replace the live `SMC_GATE_MODE=required` smoke.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def require(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise SystemExit(f"missing {label}: {needle}")


def require_order(source: str, before: str, after: str, label: str) -> None:
    left = source.find(before)
    right = source.find(after)
    if left < 0 or right < 0 or left >= right:
        raise SystemExit(f"bad order for {label}: {before!r} must precede {after!r}")


def main() -> None:
    smc = (REPO_ROOT / "tensorrt_llm" / "_torch" / "speculative" / "smc.py").read_text()
    smoke = (REPO_ROOT / "deploy" / "disagg_pd_r20" / "smoke_request_pinning.sh").read_text()
    offline = (REPO_ROOT / "deploy" / "disagg_pd_r20" / "offline_request_pinning_smoke.py").read_text()

    for needle in [
        "validate_smc_decode_request_pin",
        "TRTLLM_SMC_REQUIRE_REQUEST_PIN",
        "is_generation_only_request",
        "request_type == \"generation_only\"",
        "disagg_request_id",
        "ctx_dp_rank",
        "ctx_info_endpoint",
        "SMC-SD decode requires request pin metadata",
        "sample_state.sampler_event.synchronize()",
        "sample_state.host.new_tokens",
        "used_pinned_host_tokens = True",
        "TRTLLM_SMC_ALLOW_UNPINNED_DRAFT_COMMIT",
        "SMC-SD Moondream decode requires evented pinned host",
        "draft_token_log_probs",
        "py_smc_draft_token_log_probs",
        "SMC Moondream decode handoff preserved",
        "pinned_host_tokens=",
    ]:
        require(smc, needle, "SMC decode pinning source marker")

    require_order(
        smc,
        "validate_smc_decode_request_pin(target_model_req)",
        "target_model_req.py_draft_tokens = []",
        "request pin validation before draft-token commit",
    )
    require_order(
        smc,
        "sample_state.sampler_event.synchronize()",
        "draft_tokens_host = sample_state.host.new_tokens",
        "sampler event before pinned host token read",
    )

    for needle in [
        "SMC_GATE_MODE",
        "SMC-SD is required but decode handoff markers are missing",
        "SMC Moondream decode handoff preserved",
        "draft_token_log_probs",
        "sample_state.sampler_event",
        "pinned_host_tokens=True",
        "ctx_dp_rank=",
        "ctx_info_endpoint=",
        "ctx_dp_rank=None",
        "ctx_info_endpoint=(?:None|null|$)",
    ]:
        require(smoke, needle, "live smoke SMC-required gate")

    for needle in [
        "missing_smc_ctx_dp_rank",
        "missing_smc_ctx_info_endpoint",
        "SMC Moondream decode handoff preserved",
        "pinned_host_tokens=True",
        "positive KV transfer",
        "missing_abort_cleanup",
    ]:
        require(offline, needle, "offline fixture negative coverage")

    print("PASS static SMC-SD Moondream decode pinning audit")


if __name__ == "__main__":
    main()

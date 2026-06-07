#!/usr/bin/env python3
"""Static audit for C++ completed-prefill endpoint stamping.

The R20 request-pinning gate requires the completed prefill response to carry a
real ctx_info_endpoint. Python transceiver paths already expose their rank-info
endpoint; the C++ transceiver path must stamp its CommState identity into
ContextPhaseParams so Dynamo can pin decode to the exact prefill worker/DP rank
that produced the KV transfer.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def require(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise SystemExit(f"missing {label}: {needle}")


def main() -> None:
    cache_transceiver = (
        REPO_ROOT / "cpp" / "tensorrt_llm" / "batch_manager" / "cacheTransceiver.cpp"
    ).read_text()
    result_py = (REPO_ROOT / "tensorrt_llm" / "executor" / "result.py").read_text()

    for needle in [
        "std::optional<std::string> disaggInfoEndpoint",
        "mCommState->toString()",
        "if (!commEndpoint.empty())",
        "disaggInfoEndpoint = std::move(commEndpoint)",
    ]:
        require(cache_transceiver, needle, "C++ endpoint source marker")

    if cache_transceiver.count("disaggInfoEndpoint") < 4:
        raise SystemExit("C++ context endpoint is not threaded through both ContextPhaseParams paths")

    for needle in [
        "context_phase_params.disagg_info_endpoint",
        "ctx_info_endpoint=context_phase_params.disagg_info_endpoint",
    ]:
        require(result_py, needle, "Python result endpoint propagation")

    print("PASS static C++ completed-prefill endpoint audit")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline request-pinning/Moondream smoke parser fixture.

This is intentionally no-traffic: it validates the same proof shape required by
smoke_request_pinning.sh without touching a live canary.  It catches false
positives where the router selects workers but request pin lifecycle metadata,
positive KV transfer, abort cleanup, or SMC overlap payload preservation is
missing or inconsistent.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Fixture:
    frontend: str
    prefill: str
    decode: str
    response: dict[str, Any]
    metrics: list[dict[str, Any]]


def _base_fixture() -> Fixture:
    rid = "req-pin-123"
    frontend = "\n".join(
        [
            "Selected worker: worker_type=prefill, worker_id=0 dp_rank=0",
            f"dynamo request pin route selected request_id={rid} worker_id=0 dp_rank=0 phase=Prefill",
            f"dynamo disagg request pin established request_id={rid} disagg_request_id={rid} prefill_worker_id=0 prefill_dp_rank=Some(0) ctx_info_endpoint=nixl://ctx/0 handoff_mode=completed_prefill",
            f"dynamo disagg request pin outbound to decode request_id={rid} disagg_request_id={rid} ctx_info_endpoint=nixl://ctx/0 ctx_dp_rank=0 handoff_mode=completed_prefill",
            "Selected worker: worker_type=decode, worker_id=1 dp_rank=1",
            f"dynamo request pin route selected request_id={rid} worker_id=1 dp_rank=1 phase=Decode",
            f"dynamo request pin cleanup scheduled request_id={rid} reason=stream_closed",
            f"dynamo request pin cleared request_id={rid}",
        ]
    )
    prefill = "\n".join(
        [
            f"disagg request pin received request_type=context_only disagg_request_id={rid} ctx_request_id={rid} ctx_dp_rank=0 ctx_info_endpoint=nixl://ctx/0",
            "Initializing NIXL Connect layersplit_owner_local_alloc=true layersplit_transfer_backend=nixl global_layers=61 host_pinned_blocks=2 cache_state_layers=61",
            "disable_overlap_scheduler: False cp_type: LAYERSPLIT mla_latent_kv_dtype=kvarn_k2v2 mla_latent_kv_amortize=True",
        ]
    )
    decode = "\n".join(
        [
            f"disagg request pin received request_type=generation_only disagg_request_id={rid} ctx_request_id={rid} ctx_dp_rank=0 ctx_info_endpoint=nixl://ctx/0",
            "disable_overlap_scheduler: False backend: WARPDECODE allow_parallelism_backend_guard=false mla_latent_kv_dtype=kvarn_k2v2",
            f"SMC Moondream decode handoff preserved draft_token_log_probs sample_state.sampler_event pinned_host_tokens=True request_id={rid} disagg_request_id={rid} ctx_dp_rank=0 ctx_info_endpoint=nixl://ctx/0",
        ]
    )
    response = {
        "id": rid,
        "choices": [{"text": "one two three four five"}],
        "nvext": {
            "worker_id": {
                "prefill_worker_id": 0,
                "prefill_dp_rank": 0,
                "decode_worker_id": 1,
                "decode_dp_rank": 1,
            }
        },
    }
    metrics = [
        {
            "timing_metrics": {
                "kv_cache_size": 8192,
                "kv_cache_transfer_start": 100.0,
                "kv_cache_transfer_end": 101.5,
            }
        }
    ]
    return Fixture(frontend, prefill, decode, response, metrics)


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)


def validate(fixture: Fixture, *, smc_required: bool = True) -> None:
    all_logs = "\n".join([fixture.frontend, fixture.prefill, fixture.decode])
    bad = [
        r"Request pinning requires",
        r"ctx_dp_rank is None",
        r"SMC Moondream decode handoff preserved.*ctx_dp_rank=None",
        r"SMC Moondream decode handoff preserved.*ctx_info_endpoint=(?:None|null|$)",
        r"could not resolve dp_rank",
        r"NoBootstrapEndpoint",
        r"Disable overlap scheduler.*SMC",
        r"cp_type:\s*HELIX",
        r"\bHELIX\b.*fallback",
        r"backend: UCX",
        r"layersplit_transfer_backend: ucx",
        r"WarpDecode.*fallback",
        r"LayerSplit: layer .* no local KV pool slot",
        r"no scratch routing",
        r"host_pinned_blocks[=: ]+0\b",
        r"cache_state_layers[=: ]+0\b",
        r"pinned KV handoff.*0 blocks",
        r"KV cache transfer timeout",
        r"illegal memory access",
        r"Traceback",
    ]
    for pattern in bad:
        if re.search(pattern, all_logs, re.IGNORECASE):
            raise AssertionError(f"bad log pattern present: {pattern}")

    worker = fixture.response.get("nvext", {}).get("worker_id")
    if not isinstance(worker, dict):
        raise AssertionError("response missing nvext.worker_id")
    for key in ("prefill_worker_id", "prefill_dp_rank", "decode_worker_id", "decode_dp_rank"):
        if worker.get(key) is None:
            raise AssertionError(f"response worker metadata missing {key}")

    route_selected = re.findall(
        r"dynamo request pin route selected.*request_id[= ]([^, ]+).*worker_id[= ](\d+).*dp_rank[= ](\d+).*phase[= ](Prefill|Decode|Aggregated)",
        fixture.frontend,
    )
    established = [
        (rid, worker_id, dp_rank, "bootstrap", f"{host}:{port}")
        for rid, worker_id, dp_rank, host, port in re.findall(
            r"dynamo disagg request pin established.*request_id[= ]([^, ]+).*prefill_worker_id[= ](\d+).*prefill_dp_rank[= ](?:Some\()?(\d+).*bootstrap_host[= ]([^, ]+).*bootstrap_port[= ](\d+)",
            fixture.frontend,
        )
    ]
    established.extend(
        (rid, worker_id, dp_rank, "completed_prefill", ctx_info_endpoint)
        for rid, worker_id, dp_rank, ctx_info_endpoint in re.findall(
            r"dynamo disagg request pin established.*request_id[= ]([^, ]+).*prefill_worker_id[= ](\d+).*prefill_dp_rank[= ](?:Some\()?(\d+).*ctx_info_endpoint[= ]([^, ]+).*handoff_mode[= ]completed_prefill",
            fixture.frontend,
        )
    )
    outbound = [
        (rid, "bootstrap", f"{host}:{port}")
        for rid, host, port in re.findall(
            r"dynamo disagg request pin outbound to decode.*request_id[= ]([^, ]+).*bootstrap_host[= ]([^, ]+).*bootstrap_port[= ](\d+)",
            fixture.frontend,
        )
    ]
    outbound.extend(
        (rid, "completed_prefill", ctx_info_endpoint)
        for rid, ctx_info_endpoint in re.findall(
            r"dynamo disagg request pin outbound to decode.*request_id[= ]([^, ]+).*ctx_info_endpoint[= ]([^, ]+).*handoff_mode[= ]completed_prefill",
            fixture.frontend,
        )
    )
    cleared = re.findall(r"dynamo request pin cleared|disagg request pin cleared", all_logs)
    cleanup_scheduled = re.findall(r"dynamo request pin cleanup scheduled", all_logs)
    if not established:
        raise AssertionError("missing pin-established marker")
    if not outbound:
        raise AssertionError("missing outbound-to-decode marker")
    placeholder_completed = [
        item for item in [*established, *outbound]
        if item[-2] == "completed_prefill" and item[-1] in {"", "completed_prefill", "None", "null"}
    ]
    if placeholder_completed:
        raise AssertionError("completed-prefill pin marker missing real ctx_info_endpoint")
    shared_pin_rids = {rid for rid, *_ in established} & {rid for rid, *_ in outbound}
    if not shared_pin_rids:
        raise AssertionError("no request id shared by established and outbound markers")
    route_prefill_rids = {rid for rid, _w, _r, phase in route_selected if phase == "Prefill"}
    route_decode_rids = {rid for rid, _w, _r, phase in route_selected if phase == "Decode"}
    if not shared_pin_rids & route_prefill_rids:
        raise AssertionError("pin lifecycle does not match route-selected prefill request id")
    if not shared_pin_rids & route_decode_rids:
        raise AssertionError("pin lifecycle does not match route-selected decode request id")
    if len(cleared) < len(established):
        raise AssertionError("pin clear count is lower than established count")
    if not cleanup_scheduled:
        raise AssertionError("missing abort/early-close cleanup marker")

    positive_transfer = []
    for item in _walk(fixture.metrics):
        timing = item.get("timing_metrics") if isinstance(item, dict) else None
        if not isinstance(timing, dict):
            continue
        size = float(timing.get("kv_cache_size", 0) or 0)
        start = float(timing.get("kv_cache_transfer_start", 0) or 0)
        end = float(timing.get("kv_cache_transfer_end", 0) or 0)
        if size > 0 and start > 0 and end >= start:
            positive_transfer.append((size, start, end))
    if not positive_transfer:
        raise AssertionError("missing positive KV transfer metrics")

    if smc_required:
        for token in ("SMC Moondream decode handoff preserved", "draft_token_log_probs", "sample_state.sampler_event", "pinned_host_tokens=True", "ctx_dp_rank=", "ctx_info_endpoint="):
            if token not in all_logs:
                raise AssertionError(f"missing SMC/Moondream decode handoff marker: {token}")


def expect_failure(name: str, fixture: Fixture, needle: str | None = None) -> None:
    try:
        validate(fixture)
    except AssertionError as exc:
        if needle and needle not in str(exc):
            raise AssertionError(f"{name} failed for wrong reason: {exc}") from exc
        print(f"PASS negative {name}: {exc}")
        return
    raise AssertionError(f"negative fixture unexpectedly passed: {name}")


def main() -> None:
    good = _base_fixture()
    validate(good)
    print("PASS positive pinned NIXL + Moondream/SMC fixture")

    bad = copy.deepcopy(good)
    bad.frontend = bad.frontend.replace("dynamo request pin route selected request_id=req-pin-123 worker_id=1", "dynamo request pin route selected request_id=req-other worker_id=1", 1)
    expect_failure("decode_request_id_mismatch", bad, "route-selected decode")

    bad = copy.deepcopy(good)
    bad.metrics[0]["timing_metrics"]["kv_cache_size"] = 0
    expect_failure("zero_transfer", bad, "positive KV transfer")

    bad = copy.deepcopy(good)
    bad.frontend = bad.frontend.replace("dynamo request pin cleanup scheduled", "dynamo request pin cleanup omitted")
    expect_failure("missing_abort_cleanup", bad, "cleanup")

    bad = copy.deepcopy(good)
    bad.decode = bad.decode.replace("draft_token_log_probs sample_state.sampler_event pinned_host_tokens", "generic draft_logits greedy_fallback")
    expect_failure("missing_smc_payload", bad, "draft_token_log_probs")

    bad = copy.deepcopy(good)
    bad.decode = bad.decode.replace("ctx_dp_rank=0", "ctx_dp_rank=None")
    expect_failure("missing_smc_ctx_dp_rank", bad, "ctx_dp_rank")

    bad = copy.deepcopy(good)
    bad.decode = bad.decode.replace("ctx_info_endpoint=nixl://ctx/0", "ctx_info_endpoint=None")
    expect_failure("missing_smc_ctx_info_endpoint", bad, "ctx_info_endpoint")

    bad = copy.deepcopy(good)
    bad.decode = bad.decode.replace("pinned_host_tokens=True", "pinned_host_tokens=False")
    expect_failure("missing_smc_pinned_host_tokens", bad, "pinned_host_tokens=True")

    bad = copy.deepcopy(good)
    bad.prefill += "\nhost_pinned_blocks=0"
    expect_failure("zero_host_pinned_blocks", bad, "host_pinned_blocks")

    print(json.dumps({"offline_request_pinning_smoke": "passed"}))


if __name__ == "__main__":
    main()

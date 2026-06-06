#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Production gate harness for SMC-SD/GQA KVarN.

This script is intentionally conservative. It is safe to run with ``--dry-run``
on non-GPU hosts to print the required validation matrix. Without ``--dry-run``
it imports torch, requires the fused GQA KVarN ops, and refuses to continue
unless ``torch.ops.trtllm.kvarn_gqa_backend_ready()`` reports true. It does not
promote the reference backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SEQ_LENS = (1024, 8192, 32768, 65536, 131072)
DEFAULT_ODD_M = (1, 5, 25)
DEFAULT_TRANSPORTS = ("ucx", "nixl", "mooncake", "mori")
DEFAULT_RUNTIME_DTYPES = ("fp16", "bf16")
DEFAULT_KV_LAYOUTS = ("compact", "paged")
DEFAULT_PARTIAL_BLOCKS = (
    {"sink_tokens": 0, "tail_tokens": 0},
    {"sink_tokens": 16, "tail_tokens": 7},
    {"sink_tokens": 128, "tail_tokens": 1},
    {"sink_tokens": 128, "tail_tokens": 127},
)


@dataclass(frozen=True)
class GateCase:
    seq_len: int
    concurrency: int
    odd_m: int
    layersplit: bool
    request_pinning: bool
    moondream_overlap: bool
    smc: bool
    warpdecode: bool
    transport: str


def build_matrix(seq_lens: Iterable[int], odd_m: Iterable[int], transports: Iterable[str], concurrency: int) -> list[GateCase]:
    cases: list[GateCase] = []
    for seq_len in seq_lens:
        for m in odd_m:
            for transport in transports:
                cases.append(GateCase(
                    seq_len=seq_len,
                    concurrency=concurrency,
                    odd_m=m,
                    layersplit=True,
                    request_pinning=True,
                    moondream_overlap=True,
                    smc=True,
                    warpdecode=True,
                    transport=transport,
                ))
    return cases


def build_microbench_commands() -> list[str]:
    commands: list[str] = []
    for runtime_dtype in DEFAULT_RUNTIME_DTYPES:
        for partial in DEFAULT_PARTIAL_BLOCKS:
            commands.append(
                "python benchmarks/python/bench_kvarn_gqa_micro.py "
                "--device cuda "
                f"--runtime-dtype {runtime_dtype} "
                "--kv-heads 8 --iters 100 --sinkhorn-iters 16 "
                "--layouts compact paged --queries 1 5 25 "
                f"--sink-side-tokens {partial['sink_tokens']} "
                f"--tail-side-tokens {partial['tail_tokens']} "
                "--try-store-op --try-decode-op --try-side-op")
    return commands


def build_payload(seq_lens: Iterable[int], odd_m: Iterable[int], transports: Iterable[str], concurrency: int,
                  min_tok_s_per_user: float) -> dict[str, object]:
    cases = build_matrix(seq_lens, odd_m, transports, concurrency)
    return {
        "dtype": "kvarn_k2v2_g128",
        "dense_mla_dtype": "kvarn_k2v2",
        "dense_mla_amortize": True,
        "indexer_quantized_by_kvarn": False,
        "target_concurrency": concurrency,
        "min_tok_s_per_user": min_tok_s_per_user,
        "runtime_dtypes": list(DEFAULT_RUNTIME_DTYPES),
        "kv_layouts": list(DEFAULT_KV_LAYOUTS),
        "partial_block_cases": list(DEFAULT_PARTIAL_BLOCKS),
        "bf16_reference_required": True,
        "paged_kv_required": True,
        "abort_reuse_required": True,
        "side_pool_release_guard": True,
        "next_gpu_window_commands": build_microbench_commands(),
        "cases": [asdict(case) for case in cases],
    }


def _write_json(path: str | None, payload: object) -> None:
    if path is None:
        return
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_fused_ready() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for non-dry-run GQA KVarN production gates") from exc

    trtllm_ops = getattr(torch.ops, "trtllm", None)
    missing = []
    for name in ("kvarn_gqa_store", "kvarn_gqa_decode", "kvarn_gqa_backend_ready"):
        if trtllm_ops is None or not hasattr(trtllm_ops, name):
            missing.append(name)
    if missing:
        raise SystemExit(
            "GQA KVarN fused backend is not registered; missing torch.ops.trtllm."
            + ", torch.ops.trtllm.".join(missing))
    try:
        ready = bool(trtllm_ops.kvarn_gqa_backend_ready())
    except Exception as exc:  # pragma: no cover - exercised in integration images
        raise SystemExit(f"kvarn_gqa_backend_ready() raised: {exc}") from exc
    if not ready:
        raise SystemExit(
            "GQA KVarN fused backend is registered but not production-ready. "
            "Keep kv_cache_dtype=kvarn_k2v2_g128 fail-closed until fused store/decode, "
            "transfer, CUDA graph lifecycle, correctness, and performance gates pass.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print/emit the required gate matrix without importing torch")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=list(DEFAULT_SEQ_LENS))
    parser.add_argument("--odd-m", type=int, nargs="+", default=list(DEFAULT_ODD_M))
    parser.add_argument("--transports", nargs="+", default=list(DEFAULT_TRANSPORTS),
                        choices=list(DEFAULT_TRANSPORTS))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--min-tok-s-per-user", type=float, default=150.0)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    payload = build_payload(args.seq_lens, args.odd_m, args.transports, args.concurrency, args.min_tok_s_per_user)
    _write_json(args.json_output, payload)

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    _require_fused_ready()
    raise SystemExit(
        "Fused backend readiness passed, but deployment measurement wiring is not implemented in this harness yet. "
        "Run the SMC-SD deployment benchmark for every emitted case and record tok/s/user after first token.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"unexpected gate failure: {exc}", file=sys.stderr)
        raise

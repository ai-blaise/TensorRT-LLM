#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarize Blaise TileRT probe artifacts against a tok/s target.

The probe directory contains both real native measurements and debug scaffolds.
This report keeps those categories separate so invalid c2/c4 cache-swap bridge
numbers do not get mistaken for native serving concurrency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _uses_serial_cache_copies(record: JsonDict) -> bool:
    copy_mode = record.get("serial_context_copy_mode")
    if copy_mode not in (None, "none", "native-single"):
        return True
    return _as_int(record.get("cache_bytes_per_logical_request")) > 0


def _infer_semantics(path: Path, mode: str, record: JsonDict) -> tuple[str, bool]:
    semantics = record.get("measurement_semantics")
    valid = record.get("serving_concurrency_valid")
    if isinstance(semantics, str) and isinstance(valid, bool):
        if mode == "blaise-stream-serial-context-sweep" and _uses_serial_cache_copies(record):
            return "serial_cache_swap_debug", False
        return semantics, valid

    concurrency = _as_int(record.get("concurrency"), 1)
    if mode == "blaise-stream-serial-context-sweep":
        if concurrency == 1 and not _uses_serial_cache_copies(record):
            return "native_single_request", True
        return "serial_cache_swap_debug", False
    if mode == "blaise-stream-seqlen-sweep":
        return "native_sequence_lanes", False
    if mode == "blaise-stream-packed-concurrency-sweep":
        return "packed_sequence_lanes", False
    if "serial_context" in path.name and concurrency > 1:
        return "serial_cache_swap_debug", False
    return "native_batch", True


def _records(results_dir: Path, pattern: str) -> list[JsonDict]:
    rows = []
    for path in sorted(results_dir.glob(pattern)):
        data = _load_json(path)
        mode = str(data.get("mode", ""))
        for record in data.get("sweep") or []:
            if not isinstance(record, dict):
                continue
            semantics, valid = _infer_semantics(path, mode, record)
            row = dict(record)
            row["source"] = path.name
            row["mode"] = mode
            row["measurement_semantics"] = semantics
            row["serving_concurrency_valid"] = valid
            rows.append(row)
    return rows


def _tok_s(record: JsonDict) -> float:
    for key in ("tok_s_user_p50", "effective_tok_s_user"):
        value = _as_float(record.get(key))
        if value > 0.0:
            return value
    p50 = _as_float(record.get("forward_p50_s"))
    if p50 <= 0.0:
        return 0.0
    accepted = _as_float(record.get("mtp_accepted_p50"), 1.0)
    return accepted / p50


def _best_record(records: list[JsonDict], semantics: set[str]) -> JsonDict | None:
    candidates = [
        record
        for record in records
        if record.get("serving_concurrency_valid") is True
        and record.get("measurement_semantics") in semantics
    ]
    if not candidates:
        return None
    return max(candidates, key=_tok_s)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f}"


def _format_tok_s(value: float) -> str:
    return f"{value:.2f}"


def _print_record(prefix: str, record: JsonDict | None) -> None:
    if record is None:
        print(f"{prefix}: none")
        return
    p50 = _as_float(record.get("forward_p50_s"))
    print(
        f"{prefix}: {_format_tok_s(_tok_s(record))} tok/s/user, "
        f"p50={_format_ms(p50)} ms, "
        f"semantics={record.get('measurement_semantics')}, "
        f"source={record.get('source')}"
    )


def _scalar_gap(best_scalar: JsonDict | None, target_tok_s: float) -> None:
    if best_scalar is None:
        print("- Scalar native gap: no valid scalar record found")
        return
    p50 = _as_float(best_scalar.get("forward_p50_s"))
    if p50 <= 0.0:
        print("- Scalar native gap: missing p50")
        return
    target_seconds = 1.0 / target_tok_s
    reduction_ms = max(0.0, p50 - target_seconds) * 1000.0
    reduction_pct = max(0.0, (p50 - target_seconds) / p50) * 100.0
    print(
        "- Scalar native gap: "
        f"need p50 <= {_format_ms(target_seconds)} ms; "
        f"best is {_format_ms(p50)} ms, "
        f"cut needed={reduction_ms:.3f} ms ({reduction_pct:.2f}%)"
    )


def _mtp_gap(best_mtp: JsonDict | None, target_tok_s: float) -> None:
    if best_mtp is None:
        print("- MTP acceptance gap: no valid MTP capacity record found")
        return
    p50 = _as_float(best_mtp.get("forward_p50_s"))
    if p50 <= 0.0:
        print("- MTP acceptance gap: missing p50")
        return
    accepted = _as_float(best_mtp.get("mtp_accepted_p50"), 1.0)
    required = target_tok_s * p50
    required_integer = math.ceil(required)
    seq_len = _as_int(best_mtp.get("mtp_seq_len"), _as_int(best_mtp.get("forward_seq_len"), 1))
    raw_capacity = seq_len / p50
    print(
        "- MTP acceptance gap: "
        f"accepted_p50={accepted:.2f}, "
        f"need >= {required:.2f} ({required_integer} integer tokens) at "
        f"p50={_format_ms(p50)} ms; "
        f"raw all-accepted capacity={_format_tok_s(raw_capacity)} tok/s/user"
    )


def _invalid_summary(records: list[JsonDict]) -> None:
    invalid = [record for record in records if record.get("serving_concurrency_valid") is False]
    if not invalid:
        print("- Invalid/debug records: none")
        return
    by_semantics: dict[str, int] = {}
    for record in invalid:
        semantics = str(record.get("measurement_semantics"))
        by_semantics[semantics] = by_semantics.get(semantics, 0) + 1
    details = ", ".join(f"{key}={value}" for key, value in sorted(by_semantics.items()))
    print(f"- Invalid/debug records excluded from target proof: {len(invalid)} ({details})")


def _write_json(path: Path, records: list[JsonDict], target_tok_s: float) -> None:
    best_scalar = _best_record(
        records,
        {"native_single_request", "native_single_request_maxseq_sweep", "native_batch"},
    )
    best_mtp = _best_record(records, {"mtp_capacity_graft"})
    result = {
        "target_tok_s_user": target_tok_s,
        "record_count": len(records),
        "best_scalar": best_scalar,
        "best_mtp": best_mtp,
        "invalid_record_count": sum(
            1 for record in records if record.get("serving_concurrency_valid") is False
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".bench_runs_claude/results"),
        help="Directory containing TileRT probe JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="*tilert_blaise*.json",
        help="Glob pattern for probe JSON files inside --results-dir.",
    )
    parser.add_argument(
        "--target-tok-s",
        type=float,
        default=300.0,
        help="Target tok/s/user threshold.",
    )
    parser.add_argument("--write-json", type=Path, help="Optional machine-readable report path.")
    args = parser.parse_args()

    records = _records(args.results_dir, args.pattern)
    best_scalar = _best_record(
        records,
        {"native_single_request", "native_single_request_maxseq_sweep", "native_batch"},
    )
    best_mtp = _best_record(records, {"mtp_capacity_graft"})

    print("# Blaise TileRT result report")
    print(f"- Results: `{args.results_dir}` / `{args.pattern}`")
    print(f"- Target: {_format_tok_s(args.target_tok_s)} tok/s/user")
    print(f"- Records parsed: {len(records)}")
    _print_record("- Best valid scalar/native", best_scalar)
    _print_record("- Best valid MTP capacity", best_mtp)
    _scalar_gap(best_scalar, args.target_tok_s)
    _mtp_gap(best_mtp, args.target_tok_s)
    _invalid_summary(records)

    if args.write_json is not None:
        _write_json(args.write_json, records, args.target_tok_s)


if __name__ == "__main__":
    main()

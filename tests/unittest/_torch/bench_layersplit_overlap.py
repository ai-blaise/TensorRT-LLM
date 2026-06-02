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
"""LayerSplit overlap benchmark — M9 measurement scaffold.

Times three modes per (payload_bytes, compute_window_us) combination:

- ``M5`` — synchronous per-layer broadcast (M5c posture); each layer's
  broadcast blocks before the simulated compute starts.
- ``M6`` — single-channel cross-layer prefetch (M6 posture); the next
  layer's broadcast runs on ``comm_stream`` while the current layer's
  simulated compute runs on the default stream.
- ``M8b`` — 2-channel cross-layer prefetch (M8b posture); the indexer
  channel uses ``indexer_comm_stream`` and the dense KV channel uses
  ``comm_stream``, both prefetched in parallel.

A child process per CP rank initializes NCCL, drives the chosen mode
across ``num_layers`` iterations with a configurable payload size and a
configurable per-layer ``cuda._sleep`` simulating the indexer + sparse
attention compute. The runner reports the wall-clock total per mode so
the overlap wins are visible as smaller wall times for M6 / M8b vs M5.

NOT a unit test — does no correctness checking (M5b / M6 / M8b multi-proc
tests already cover that). Drives only the timing baseline that the M5d
real-payload work and any subsequent optimization can compare against.

Invoke via ``bench_runner_layersplit_overlap.py`` (a small driver that
spawns the two ranks); the worker here is exported only as a callable
for that runner.
"""
import os
import time
from typing import List, Tuple

import torch
import torch.distributed as dist


def _simulate_compute_us(stream: torch.cuda.Stream, us: int) -> None:
    """Issue a CUDA-side sleep of ``us`` microseconds on ``stream``.

    ``torch.cuda._sleep(cycles)`` blocks the stream for ``cycles`` GPU
    cycles. The conversion to microseconds is the device's clock rate;
    NVIDIA Blackwell B200 nominally runs at 1.59 GHz so 1 us is roughly
    1590 cycles. We use 1500 as a conservative round number — the exact
    rate doesn't matter for relative comparisons across modes since
    every mode pays the same simulated compute.
    """
    if us <= 0:
        return
    with torch.cuda.stream(stream):
        torch.cuda._sleep(int(us * 1500))


def _bench_mode_m5(rank: int, world_size: int, num_layers: int,
                   payload_bytes: int, compute_us: int, state,
                   cp_group) -> float:
    """Synchronous per-layer broadcast (M5c posture)."""
    payload = torch.zeros(payload_bytes, dtype=torch.uint8, device="cuda")
    default_stream = torch.cuda.current_stream()
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for layer_idx in range(num_layers):
        state.maybe_broadcast_for_layer(
            layer_idx=layer_idx,
            payload=payload,
            cp_group=cp_group,
            async_op=False,
            channel="kv",
        )
        _simulate_compute_us(default_stream, compute_us)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_mode_m6(rank: int, world_size: int, num_layers: int,
                   payload_bytes: int, compute_us: int, state,
                   cp_group) -> float:
    """Single-channel cross-layer prefetch (M6 posture)."""
    payloads = [
        torch.zeros(payload_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_layers)
    ]
    default_stream = torch.cuda.current_stream()
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Bootstrap layer 0
    state.maybe_broadcast_for_layer(layer_idx=0,
                                    payload=payloads[0],
                                    cp_group=cp_group,
                                    async_op=False,
                                    channel="kv")
    if num_layers >= 2:
        state.prefetch_for_layer(layer_idx=1,
                                 payload=payloads[1],
                                 cp_group=cp_group,
                                 channel="kv")
    _simulate_compute_us(default_stream, compute_us)
    for layer_idx in range(1, num_layers):
        state.wait_for_prefetched_layer(layer_idx, channel="kv")
        next_layer = layer_idx + 1
        if next_layer < num_layers:
            state.prefetch_for_layer(layer_idx=next_layer,
                                     payload=payloads[next_layer],
                                     cp_group=cp_group,
                                     channel="kv")
        _simulate_compute_us(default_stream, compute_us)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_mode_m8b(rank: int, world_size: int, num_layers: int,
                    payload_bytes: int, compute_us: int, state,
                    cp_group) -> float:
    """2-channel cross-layer prefetch (M8b posture)."""
    # The "indexer" channel publishes ~1/8 the bytes of the "kv" channel.
    # We approximate that ratio by splitting payload_bytes as
    # indexer = max(16, payload_bytes // 8), kv = payload_bytes - indexer.
    indexer_bytes = max(16, payload_bytes // 8)
    kv_bytes = max(16, payload_bytes - indexer_bytes)
    indexer_payloads = [
        torch.zeros(indexer_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_layers)
    ]
    kv_payloads = [
        torch.zeros(kv_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_layers)
    ]
    default_stream = torch.cuda.current_stream()
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Bootstrap layer 0 on both channels
    for channel, p in (("indexer", indexer_payloads[0]), ("kv",
                                                          kv_payloads[0])):
        state.maybe_broadcast_for_layer(layer_idx=0,
                                        payload=p,
                                        cp_group=cp_group,
                                        async_op=False,
                                        channel=channel)
    if num_layers >= 2:
        for channel, p in (("indexer", indexer_payloads[1]),
                           ("kv", kv_payloads[1])):
            state.prefetch_for_layer(layer_idx=1,
                                     payload=p,
                                     cp_group=cp_group,
                                     channel=channel)
    _simulate_compute_us(default_stream, compute_us)
    for layer_idx in range(1, num_layers):
        for channel in ("indexer", "kv"):
            state.wait_for_prefetched_layer(layer_idx, channel=channel)
        next_layer = layer_idx + 1
        if next_layer < num_layers:
            for channel, payloads in (("indexer", indexer_payloads),
                                      ("kv", kv_payloads)):
                state.prefetch_for_layer(layer_idx=next_layer,
                                         payload=payloads[next_layer],
                                         cp_group=cp_group,
                                         channel=channel)
        _simulate_compute_us(default_stream, compute_us)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_worker(rank: int,
                  world_size: int,
                  master_port: int,
                  num_layers: int,
                  payload_bytes: int,
                  compute_us: int,
                  warmup_iters: int,
                  measure_iters: int,
                  result_path: str) -> None:
    """Per-process bench entry point."""
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=rank)
    try:
        from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
            LayerSplitRuntimeState)
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            layersplit_enabled=True,
            layersplit_owner_assignment="round_robin",
            layersplit_transfer_backend="auto",
            layersplit_all_cp_ranks_transfer=True,
        )
        state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=cfg,
            num_layers=num_layers,
            cp_size=world_size,
            cp_rank=rank,
            create_comm_stream=True,
            create_indexer_comm_stream=True,
        )
        state.bind_cp_group(dist.group.WORLD)

        modes = [("M5", _bench_mode_m5), ("M6", _bench_mode_m6),
                 ("M8b", _bench_mode_m8b)]
        results: List[Tuple[str, float, float, float]] = []
        for mode_name, mode_fn in modes:
            # Reset prefetch state between modes
            state.clear_prefetched_events()
            # Warm up
            for _ in range(warmup_iters):
                mode_fn(rank, world_size, num_layers, payload_bytes,
                        compute_us, state, dist.group.WORLD)
                state.clear_prefetched_events()
            # Measure
            samples = []
            for _ in range(measure_iters):
                t = mode_fn(rank, world_size, num_layers, payload_bytes,
                            compute_us, state, dist.group.WORLD)
                state.clear_prefetched_events()
                samples.append(t)
            samples.sort()
            results.append((
                mode_name,
                samples[0],
                sum(samples) / len(samples),
                samples[-1],
            ))

        # Rank 0 writes the report.
        if rank == 0:
            with open(result_path, "w") as f:
                f.write(f"num_layers={num_layers} payload_bytes="
                        f"{payload_bytes} compute_us={compute_us} "
                        f"warmup={warmup_iters} measure={measure_iters}\n")
                f.write(f"{'mode':5s} {'min_ms':>10s} {'mean_ms':>10s} "
                        f"{'max_ms':>10s}\n")
                for mode_name, mn, mean, mx in results:
                    f.write(f"{mode_name:5s} {mn*1e3:10.4f} {mean*1e3:10.4f} "
                            f"{mx*1e3:10.4f}\n")
                # Overlap savings: M6 vs M5, M8b vs M5
                m5_mean = next(r[2] for r in results if r[0] == "M5")
                m6_mean = next(r[2] for r in results if r[0] == "M6")
                m8b_mean = next(r[2] for r in results if r[0] == "M8b")
                f.write(f"\noverlap_savings_vs_M5:\n")
                f.write(f"  M6 :  {(m5_mean - m6_mean)/m5_mean*100:+.2f}% "
                        f"({(m5_mean - m6_mean)*1e3:.3f} ms wall)\n")
                f.write(f"  M8b:  {(m5_mean - m8b_mean)/m5_mean*100:+.2f}% "
                        f"({(m5_mean - m8b_mean)*1e3:.3f} ms wall)\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(
        "This module is the worker; invoke via "
        "tests/unittest/_torch/run_bench_layersplit_overlap.py")

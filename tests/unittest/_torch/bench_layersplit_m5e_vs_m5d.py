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
"""LayerSplit M5e vs M5d wire-byte + wall-time bench.

Establishes the measured-on-real-NCCL win of the M5e active-block
broadcast over the M5d full-pool baseline that the production hook
shipped before this commit. Spawns two CP ranks, allocates a fake
per-layer cache pool, and measures three modes per `(num_blocks,
active_per_step)` shape:

- ``M5d_full``: ``dist.broadcast(cache_slot, src=owner)`` — broadcasts
  the WHOLE pool slot every layer. This is what M5d shipped before M5e
  fixed it; included here to quantify M5e's improvement.
- ``M5e_active``: gather + broadcast + scatter the rows referenced by
  ``active_block_ids`` only. Production today.
- ``noop``: no broadcast at all (compute baseline) — the wall-time
  delta against this isolates the LayerSplit broadcast cost.

Reports per-mode min/mean wall time and bytes-on-the-wire. The headline
metric is ``M5e wins by`` = M5d wall time / M5e wall time at the
production-realistic shape.
"""
import os
import sys
import time

import torch
import torch.distributed as dist


def _bench_m5d_full(num_layers, cache_slot, _active_ids, cp_group, src_rank):
    """Broadcast the WHOLE cache slot per layer (M5d baseline)."""
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_layers):
        dist.broadcast(cache_slot,
                       src=src_rank,
                       group=cp_group,
                       async_op=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_m5e_active(num_layers, cache_slot, active_ids, cp_group, src_rank):
    """Gather + broadcast + scatter the active block subset (M5e)."""
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_layers):
        send_buf = cache_slot.index_select(0, active_ids).contiguous()
        dist.broadcast(send_buf,
                       src=src_rank,
                       group=cp_group,
                       async_op=False)
        cache_slot.index_copy_(0, active_ids, send_buf)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_noop(num_layers, _cache_slot, _active_ids, cp_group, _src_rank):
    """Compute-only baseline — no broadcast."""
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_layers):
        pass
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _worker(rank, world_size, master_port, num_layers, num_blocks_per_layer,
            block_bytes, active_per_step, result_path):
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
        cache_slot = torch.zeros((num_blocks_per_layer, block_bytes),
                                 dtype=torch.uint8,
                                 device="cuda")
        # Deterministic per-step set of active block ids — same on every rank.
        active_ids = torch.arange(0,
                                  active_per_step,
                                  dtype=torch.int64,
                                  device="cuda")
        warmup = 3
        measure = 10
        modes = [
            ("M5d_full", _bench_m5d_full),
            ("M5e_active", _bench_m5e_active),
            ("noop", _bench_noop),
        ]
        results = []
        for name, fn in modes:
            for _ in range(warmup):
                fn(num_layers, cache_slot, active_ids, dist.group.WORLD, 0)
            samples = []
            for _ in range(measure):
                samples.append(
                    fn(num_layers, cache_slot, active_ids, dist.group.WORLD,
                       0))
            samples.sort()
            results.append((name, samples[0], sum(samples) / len(samples)))

        if rank == 0:
            full_slot_bytes = num_blocks_per_layer * block_bytes * num_layers
            active_bytes = active_per_step * block_bytes * num_layers
            with open(result_path, "w") as f:
                f.write(
                    f"num_layers={num_layers} num_blocks_per_layer="
                    f"{num_blocks_per_layer} block_bytes={block_bytes} "
                    f"active_per_step={active_per_step}\n")
                f.write(f"M5d wire bytes (full pool): "
                        f"{full_slot_bytes/1e6:.2f} MB / forward step\n")
                f.write(f"M5e wire bytes (active):    "
                        f"{active_bytes/1e6:.2f} MB / forward step  "
                        f"(reduction: {full_slot_bytes/active_bytes:.1f}x)\n")
                f.write("\n")
                f.write(f"{'mode':12s} {'min_ms':>10s} {'mean_ms':>10s}\n")
                for name, mn, mean in results:
                    f.write(f"{name:12s} {mn*1e3:10.4f} {mean*1e3:10.4f}\n")
                m5d_mean = next(r[2] for r in results if r[0] == "M5d_full")
                m5e_mean = next(r[2] for r in results if r[0] == "M5e_active")
                f.write(f"\nM5e wins by {m5d_mean/m5e_mean:.2f}x over M5d "
                        f"(wall time, mean) "
                        f"({(m5d_mean-m5e_mean)*1e3:.3f} ms saved per forward "
                        f"step on CP=2)\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(
        "This module is the worker; invoke via run_bench_layersplit_m5e_vs_m5d.py")

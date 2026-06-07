#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused KVarN GQA side-state NIXL transfer probe.

Builds tensors with the same per-layer/per-slot layout used by
_KVarNGQASidePool.transfer_meta(), transfers a nonzero request slot through the
TensorRT-LLM NIXL transfer agent, and verifies byte-for-byte receiver parity.
A failing runtime probe is blocker evidence, not a fallback path.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass
class SideEntry:
    name: str
    item_size: int
    src_ptr: int
    dst_ptr: int
    src_tensor: object
    dst_tensor: object


def _side_shapes(layers: int, slots: int, sink: int, group: int, kv_heads: int, head_dim: int):
    return {
        "sink_k": (layers, slots, sink, kv_heads, head_dim),
        "sink_v": (layers, slots, sink, kv_heads, head_dim),
        "sink_len": (layers, slots),
        "tail_k": (layers, slots, group, kv_heads, head_dim),
        "tail_v": (layers, slots, group, kv_heads, head_dim),
        "tail_filled": (layers, slots, group),
        "tail_block_start": (layers, slots),
        "committed": (layers, slots, max(1, 4)),
        "commit_gen": (layers, slots, max(1, 4)),
    }


def dry_run(args: argparse.Namespace) -> None:
    total = 0
    elems = {
        "sink_k": 2,
        "sink_v": 2,
        "sink_len": 4,
        "tail_k": 2,
        "tail_v": 2,
        "tail_filled": 1,
        "tail_block_start": 8,
        "committed": 1,
        "commit_gen": 8,
    }
    for name, shape in _side_shapes(args.layers, args.slots, args.sink_tokens, args.group,
                                    args.kv_heads, args.head_dim).items():
        per_slot_shape = shape[2:] if len(shape) > 2 else ()
        item = int(math.prod(per_slot_shape) if per_slot_shape else 1) * elems[name]
        total += item * args.layers
        print(f"SIDE_ENTRY name={name} shape={shape} per_slot_bytes={item}")
    print(
        f"KVARN_GQA_NIXL_SIDE_DRY_RUN layers={args.layers} slots={args.slots} "
        f"src_slot={args.src_slot} dst_slot={args.dst_slot} fragments={args.layers * 9} "
        f"slot_bytes={total} backend={args.backend} op={args.op} memory={args.memory}"
    )


def _make_entries(torch, args: argparse.Namespace, device) -> list[SideEntry]:
    dtypes = {
        "sink_k": torch.float16,
        "sink_v": torch.float16,
        "sink_len": torch.int32,
        "tail_k": torch.float16,
        "tail_v": torch.float16,
        "tail_filled": torch.bool,
        "tail_block_start": torch.int64,
        "committed": torch.bool,
        "commit_gen": torch.int64,
    }
    entries: list[SideEntry] = []
    for name, shape in _side_shapes(args.layers, args.slots, args.sink_tokens, args.group,
                                    args.kv_heads, args.head_dim).items():
        dtype = dtypes[name]
        src = torch.empty(shape, device=device, dtype=dtype)
        dst = torch.empty_like(src)
        if dtype is torch.bool:
            src.copy_((torch.arange(src.numel(), device=device).reshape(shape) % 2) == 0)
            dst.zero_()
        elif dtype in (torch.int32, torch.int64):
            src.copy_(torch.arange(src.numel(), device=device, dtype=dtype).reshape(shape) + 17)
            dst.zero_()
        else:
            src.copy_((torch.arange(src.numel(), device=device, dtype=torch.float32).reshape(shape) % 127).to(dtype))
            dst.zero_()
        for layer in range(args.layers):
            src_view = src[layer, args.src_slot]
            dst_view = dst[layer, args.dst_slot]
            item_size = int(src_view.numel() * src_view.element_size())
            entries.append(SideEntry(
                name=f"layer{layer}.{name}",
                item_size=item_size,
                src_ptr=int(src_view.data_ptr()),
                dst_ptr=int(dst_view.data_ptr()),
                src_tensor=src_view,
                dst_tensor=dst_view,
            ))
    return entries


def _register_descs(entries: Iterable[SideEntry], mem_type: str):
    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs

    return (
        RegMemoryDescs(mem_type, [(e.src_ptr, e.item_size, 0, f"src.{e.name}") for e in entries]),
        RegMemoryDescs(mem_type, [(e.dst_ptr, e.item_size, 0, f"dst.{e.name}") for e in entries]),
    )


def _memory_descs(entries: list[SideEntry], which: str, mem_type):
    import numpy as np
    from tensorrt_llm._torch.disaggregation.base.agent import MemoryDescs

    ptrs = np.array([getattr(e, f"{which}_ptr") for e in entries], dtype=np.int64)
    sizes = np.array([e.item_size for e in entries], dtype=np.int64)
    if hasattr(MemoryDescs, "from_arrays_uniform_device"):
        return MemoryDescs.from_arrays_uniform_device(mem_type, ptrs, sizes, 0)
    return MemoryDescs(mem_type, [(int(p), int(s), 0) for p, s in zip(ptrs, sizes)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--src-slot", type=int, default=1)
    parser.add_argument("--dst-slot", type=int, default=2)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--backend", choices=("LIBFABRIC", "UCX"), default="LIBFABRIC")
    parser.add_argument("--memory", choices=("VRAM", "DRAM"), default="VRAM")
    parser.add_argument("--op", choices=("WRITE", "READ"), default="WRITE")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.src_slot >= args.slots or args.dst_slot >= args.slots:
        raise SystemExit("src/dst slots must be less than --slots")
    if args.dry_run:
        dry_run(args)
        return

    os.environ["TRTLLM_NIXL_KVCACHE_BACKEND"] = args.backend
    os.environ.setdefault("TRTLLM_NIXL_NUM_THREADS", "0")

    import torch
    from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
    from tensorrt_llm._torch.disaggregation.base.agent import MemoryType, TransferOp, TransferRequest

    if args.memory == "VRAM":
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        mem_type = MemoryType.VRAM
    else:
        device = torch.device("cpu")
        mem_type = MemoryType.DRAM

    entries = _make_entries(torch, args, device)
    src_name = "kvarn_gqa_side_src_probe"
    dst_name = "kvarn_gqa_side_dst_probe"
    src_agent = NixlTransferAgent(src_name, True, num_threads=int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "0")))
    dst_agent = NixlTransferAgent(dst_name, True, num_threads=int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "0")))
    src_reg, dst_reg = _register_descs(entries, args.memory)
    src_agent.register_memory(src_reg)
    dst_agent.register_memory(dst_reg)
    src_agent.load_remote_agent(dst_name, dst_agent.get_local_agent_desc())
    dst_agent.load_remote_agent(src_name, src_agent.get_local_agent_desc())

    if args.op == "WRITE":
        agent = src_agent
        remote = dst_name
        op = TransferOp.WRITE
    else:
        agent = dst_agent
        remote = src_name
        op = TransferOp.READ
    status = agent.submit_transfer_requests(TransferRequest(
        op,
        _memory_descs(entries, "src", mem_type),
        _memory_descs(entries, "dst", mem_type),
        remote,
        None,
    ))
    ok = status.wait(timeout_ms=args.timeout_ms)
    if args.memory == "VRAM":
        torch.cuda.synchronize(device)
    mismatches = []
    for e in entries:
        if e.src_tensor.dtype is torch.bool:
            diff = int((e.src_tensor != e.dst_tensor).to(torch.int32).max().item())
        elif e.src_tensor.dtype.is_floating_point:
            diff = float((e.src_tensor - e.dst_tensor).abs().max().item())
        else:
            diff = int((e.src_tensor - e.dst_tensor).abs().max().item())
        if diff != 0:
            mismatches.append((e.name, diff, e.item_size))
    print(
        f"KVARN_GQA_NIXL_SIDE_RESULT backend={args.backend} memory={args.memory} op={args.op} "
        f"wait={ok} done={status.is_completed()} fragments={len(entries)} "
        f"bytes={sum(e.item_size for e in entries)} mismatches={len(mismatches)}"
    )
    if mismatches:
        print("KVARN_GQA_NIXL_SIDE_MISMATCH", mismatches[:8])
        raise SystemExit(1)
    if not ok or not status.is_completed():
        raise SystemExit(1)
    print("KVARN_GQA_NIXL_SIDE_OK")


if __name__ == "__main__":
    main()

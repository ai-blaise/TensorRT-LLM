"""Multi-process NCCL integration test for LayerSplitRuntimeState.

Runs on a pair of GPUs (default: GPUs 0 and 1 from the visible set, which
the caller maps via CUDA_VISIBLE_DEVICES). Spawns two ranks, builds a real
NCCL process group, constructs LayerSplitRuntimeState with cp_size=2, and
verifies that maybe_broadcast_for_layer publishes the owner's tensor to
the peer rank for every layer index across both owner-assignment policies.

This test requires actual GPUs and works around the NVLink SHARP
Multicast collision that occurs when another NCCL tenant (e.g. a serving
deployment) holds NVSwitch resources on the same node: it sets
NCCL_NVLS_ENABLE=0 in the child processes before initializing the group.
"""
import os
import sys

import pytest
import torch


# Skip the test cleanly when CUDA isn't present (the rest of the suite
# is CPU-friendly via the direct-importlib runner used for the other
# layersplit tests).
HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _worker(rank, world_size, master_port, policy, num_layers, tmpdir_path):
    """Child-process entry point."""
    # NVLink SHARP Multicast (NVLS) conflicts with sibling NCCL tenants on
    # B200 / GB200 nodes; disable it to coexist with a serving deployment
    # that already holds the NVSwitch resources.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    # The parent sets CUDA_VISIBLE_DEVICES so local_rank 0..world_size-1
    # always map to the available GPU pair.
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    try:
        # Late-import so child processes don't pay a CUDA init before
        # set_device.
        from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
            LayerSplitRuntimeState)
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            layersplit_enabled=True,
            layersplit_owner_assignment=policy,
            layersplit_transfer_backend="auto",
            layersplit_all_cp_ranks_transfer=True,
        )
        state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=cfg,
            num_layers=num_layers,
            cp_size=world_size,
            cp_rank=rank,
        )
        state.bind_cp_group(dist.group.WORLD)

        # For every layer, the owner publishes a distinctive payload; all
        # ranks call broadcast (NCCL requires every participant). Non-owner
        # ranks must end up with the owner's value.
        errors = []
        for layer_idx in range(num_layers):
            owner = state.ownership.owner_of(layer_idx)
            payload = torch.empty(4, dtype=torch.int32, device="cuda")
            if rank == owner:
                payload[:] = torch.tensor(
                    [layer_idx * 10 + 0, layer_idx * 10 + 1,
                     layer_idx * 10 + 2, layer_idx * 10 + 3],
                    device="cuda")
            else:
                payload[:] = -1
            state.maybe_broadcast_for_layer(
                layer_idx=layer_idx,
                payload=payload,
                cp_group=dist.group.WORLD,
                async_op=False,
            )
            torch.cuda.synchronize()
            expected = [layer_idx * 10 + i for i in range(4)]
            got = payload.tolist()
            if got != expected:
                errors.append(
                    f"layer {layer_idx} rank {rank} owner {owner}: "
                    f"got {got}, expected {expected}")

        # Record result to disk so the parent can assert it; rank 0 writes
        # the consolidated result file.
        result_path = os.path.join(tmpdir_path, f"rank{rank}.result")
        with open(result_path, "w") as f:
            if errors:
                f.write("FAIL\n")
                for e in errors:
                    f.write(e + "\n")
            else:
                f.write(f"PASS rank={rank} layers={num_layers}\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not HAS_CUDA,
                    reason="needs >=2 CUDA devices for multi-rank NCCL test")
@pytest.mark.parametrize("policy", ["round_robin", "contiguous"])
def test_real_nccl_broadcast_publishes_owner_payload(tmp_path, policy):
    """End-to-end NCCL test: 2 ranks, real broadcast, both policies."""
    import torch.multiprocessing as mp

    world_size = 2
    num_layers = 8
    master_port = 29510 + (0 if policy == "round_robin" else 1)

    mp.spawn(
        _worker,
        args=(world_size, master_port, policy, num_layers, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        result_path = tmp_path / f"rank{rank}.result"
        assert result_path.exists(), f"rank {rank} produced no result file"
        contents = result_path.read_text()
        assert contents.startswith("PASS"), (
            f"rank {rank} failed under policy={policy}:\n{contents}")


def _worker_m6(rank, world_size, master_port, policy, num_layers,
               tmpdir_path):
    """Child-process entry point for the M6 cross-layer overlap test.

    Drives the prefetch protocol: at step L, wait on the previously
    prefetched broadcast for layer L (or do a sync broadcast if layer L
    is the bootstrap step), then prefetch layer L+1 on the comm stream.
    The receiver content for layer L must match the owner's published
    payload regardless of which order the broadcasts overlap.
    """
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    try:
        from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
            LayerSplitRuntimeState)
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            layersplit_enabled=True,
            layersplit_owner_assignment=policy,
            layersplit_transfer_backend="auto",
            layersplit_all_cp_ranks_transfer=True,
        )
        state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=cfg,
            num_layers=num_layers,
            cp_size=world_size,
            cp_rank=rank,
        )
        state.bind_cp_group(dist.group.WORLD)

        # Pre-allocate per-layer payloads. On the OWNER rank for layer L
        # we stamp a layer-specific magic value; on non-owners we stamp -1.
        # Every rank participates in every broadcast (NCCL requirement).
        payloads = []
        for layer_idx in range(num_layers):
            owner = state.ownership.owner_of(layer_idx)
            p = torch.empty(4, dtype=torch.int32, device="cuda")
            if rank == owner:
                p[:] = torch.tensor(
                    [layer_idx * 100 + i for i in range(4)],
                    device="cuda")
            else:
                p[:] = -1
            payloads.append(p)
            # Also seed the state's per-layer payload dict so prefetch and
            # waitevent see the same tensor as our verification reads.
            state._per_layer_payloads[layer_idx] = p

        errors = []

        # Bootstrap: layer 0 has no in-flight prefetch -> sync broadcast.
        state.maybe_broadcast_for_layer(
            layer_idx=0,
            payload=payloads[0],
            cp_group=dist.group.WORLD,
            async_op=False,
        )
        # Prefetch layer 1
        if num_layers >= 2:
            state.prefetch_for_layer(
                layer_idx=1,
                payload=payloads[1],
                cp_group=dist.group.WORLD,
            )

        # Pipeline: for L in 1..N-1, wait on prefetched L, then prefetch L+1
        for layer_idx in range(1, num_layers):
            waited = state.wait_for_prefetched_layer(layer_idx)
            if not waited:
                # Defensive: fall back to sync broadcast if a prefetch
                # was somehow missed (shouldn't happen in this script).
                state.maybe_broadcast_for_layer(
                    layer_idx=layer_idx,
                    payload=payloads[layer_idx],
                    cp_group=dist.group.WORLD,
                    async_op=False,
                )
            next_layer = layer_idx + 1
            if next_layer < num_layers:
                state.prefetch_for_layer(
                    layer_idx=next_layer,
                    payload=payloads[next_layer],
                    cp_group=dist.group.WORLD,
                )

        # All broadcasts must have completed by now.
        torch.cuda.synchronize()

        for layer_idx in range(num_layers):
            expected = [layer_idx * 100 + i for i in range(4)]
            got = payloads[layer_idx].tolist()
            if got != expected:
                owner = state.ownership.owner_of(layer_idx)
                errors.append(
                    f"M6 layer {layer_idx} rank {rank} owner {owner}: "
                    f"got {got}, expected {expected}")

        # Sanity: every prefetched event must have been consumed.
        if state._prefetched_events:
            errors.append(
                f"M6 leftover prefetched events: "
                f"{sorted(state._prefetched_events.keys())}")

        result_path = os.path.join(tmpdir_path, f"rank{rank}.result")
        with open(result_path, "w") as f:
            if errors:
                f.write("FAIL\n")
                for e in errors:
                    f.write(e + "\n")
            else:
                f.write(
                    f"PASS rank={rank} layers={num_layers} prefetch=on\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not HAS_CUDA,
                    reason="needs >=2 CUDA devices for multi-rank NCCL test")
@pytest.mark.parametrize("policy", ["round_robin", "contiguous"])
def test_real_nccl_prefetch_pipeline_publishes_owner_payload(
        tmp_path, policy):
    """End-to-end NCCL test of the M6 cross-layer prefetch protocol."""
    import torch.multiprocessing as mp

    world_size = 2
    num_layers = 8
    master_port = 29520 + (0 if policy == "round_robin" else 1)

    mp.spawn(
        _worker_m6,
        args=(world_size, master_port, policy, num_layers, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        result_path = tmp_path / f"rank{rank}.result"
        assert result_path.exists(), f"rank {rank} produced no result file"
        contents = result_path.read_text()
        assert contents.startswith("PASS"), (
            f"rank {rank} failed under policy={policy}:\n{contents}")


def _worker_m8b(rank, world_size, master_port, policy, num_layers,
                tmpdir_path):
    """Child-process entry point for the M8b 2-channel prefetch test.

    Drives the channel-aware prefetch protocol: every layer publishes
    both an indexer-K payload (smaller) and a dense KV payload (larger).
    The two channels use independent comm streams + independent
    prefetched-event dict entries; at layer L the hook waits on both
    channels for L (popping their events) and prefetches both for L+1.
    Verifies the receiver sees the correct content for every layer on
    every channel.
    """
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    try:
        from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
            LayerSplitRuntimeState)
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            layersplit_enabled=True,
            layersplit_owner_assignment=policy,
            layersplit_transfer_backend="auto",
            layersplit_all_cp_ranks_transfer=True,
        )
        state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=cfg,
            num_layers=num_layers,
            cp_size=world_size,
            cp_rank=rank,
            # Force both streams on so the M8b two-stream path runs.
            create_comm_stream=True,
            create_indexer_comm_stream=True,
        )
        state.bind_cp_group(dist.group.WORLD)
        assert state.comm_stream is not None
        assert state.indexer_comm_stream is not None
        assert state.comm_stream is not state.indexer_comm_stream

        # Per-layer per-channel payloads. Owner stamps a channel-distinct
        # magic value; non-owner stamps -1.
        payloads = {}
        for layer_idx in range(num_layers):
            owner = state.ownership.owner_of(layer_idx)
            for ch_idx, channel in enumerate(("indexer", "kv")):
                p = torch.empty(4, dtype=torch.int32, device="cuda")
                if rank == owner:
                    # Distinct values per (layer, channel) to detect any
                    # cross-channel aliasing.
                    base = layer_idx * 1000 + ch_idx * 100
                    p[:] = torch.tensor([base + i for i in range(4)],
                                         device="cuda")
                else:
                    p[:] = -1
                payloads[(layer_idx, channel)] = p
                state._per_layer_payloads[(layer_idx, channel)] = p

        errors = []

        # Bootstrap layer 0 on both channels.
        for channel in ("indexer", "kv"):
            state.maybe_broadcast_for_layer(
                layer_idx=0,
                payload=payloads[(0, channel)],
                cp_group=dist.group.WORLD,
                async_op=False,
                channel=channel,
            )
        # Prefetch layer 1 on both channels.
        if num_layers >= 2:
            for channel in ("indexer", "kv"):
                state.prefetch_for_layer(
                    layer_idx=1,
                    payload=payloads[(1, channel)],
                    cp_group=dist.group.WORLD,
                    channel=channel,
                )

        # Pipeline.
        for layer_idx in range(1, num_layers):
            for channel in ("indexer", "kv"):
                waited = state.wait_for_prefetched_layer(layer_idx,
                                                         channel=channel)
                if not waited:
                    state.maybe_broadcast_for_layer(
                        layer_idx=layer_idx,
                        payload=payloads[(layer_idx, channel)],
                        cp_group=dist.group.WORLD,
                        async_op=False,
                        channel=channel,
                    )
            next_layer = layer_idx + 1
            if next_layer < num_layers:
                for channel in ("indexer", "kv"):
                    state.prefetch_for_layer(
                        layer_idx=next_layer,
                        payload=payloads[(next_layer, channel)],
                        cp_group=dist.group.WORLD,
                        channel=channel,
                    )

        torch.cuda.synchronize()

        for layer_idx in range(num_layers):
            for ch_idx, channel in enumerate(("indexer", "kv")):
                base = layer_idx * 1000 + ch_idx * 100
                expected = [base + i for i in range(4)]
                got = payloads[(layer_idx, channel)].tolist()
                if got != expected:
                    owner = state.ownership.owner_of(layer_idx)
                    errors.append(
                        f"M8b layer {layer_idx} channel {channel} rank "
                        f"{rank} owner {owner}: got {got}, expected "
                        f"{expected}")

        if state._prefetched_events:
            errors.append(
                f"M8b leftover prefetched events: "
                f"{sorted(state._prefetched_events.keys())}")

        result_path = os.path.join(tmpdir_path, f"rank{rank}.result")
        with open(result_path, "w") as f:
            if errors:
                f.write("FAIL\n")
                for e in errors:
                    f.write(e + "\n")
            else:
                f.write(
                    f"PASS rank={rank} layers={num_layers} channels=2 "
                    f"streams=2\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not HAS_CUDA,
                    reason="needs >=2 CUDA devices for multi-rank NCCL test")
@pytest.mark.parametrize("policy", ["round_robin", "contiguous"])
def test_real_nccl_two_channel_prefetch_pipeline(tmp_path, policy):
    """End-to-end NCCL test of the M8b 2-channel (indexer + KV) prefetch
    pipeline with independent comm streams."""
    import torch.multiprocessing as mp

    world_size = 2
    num_layers = 8
    master_port = 29530 + (0 if policy == "round_robin" else 1)

    mp.spawn(
        _worker_m8b,
        args=(world_size, master_port, policy, num_layers, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        result_path = tmp_path / f"rank{rank}.result"
        assert result_path.exists(), f"rank {rank} produced no result file"
        contents = result_path.read_text()
        assert contents.startswith("PASS"), (
            f"rank {rank} failed under policy={policy}:\n{contents}")


if __name__ == "__main__":
    # Allow standalone invocation: python -m pytest <this file> -v
    sys.exit(pytest.main([__file__, "-v", "-s"]))

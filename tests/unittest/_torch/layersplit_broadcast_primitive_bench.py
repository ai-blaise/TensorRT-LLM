"""Phase-1A SECTION-2 broadcast-primitive benchmark (the maximum-optimization
diligence).

Question (user directive 7): the per-layer LayerSplit active-block broadcast
publishes the OWNER rank's blocks to its CP peers. We carve the CP subgroup via
an MPI-TCPStore rendezvous (one-time setup) and the steady-state primitive is a
NCCL ``dist.broadcast`` on that subgroup. TRT-LLM deliberately uses a custom C++
IPC allreduce over vanilla NCCL because NCCL loses on small/medium NVLink
payloads. So: at the REAL active-block sizes, is NCCL ``dist.broadcast``
competitive with a custom IPC/NVLink P2P copy (owner -> each peer via
``cudaMemcpyPeerAsync``, expressed as a cross-device ``dst.copy_(src)`` over a
P2P-enabled path), or does IPC win?

This bench measures, on real B200, over a payload size sweep that brackets the
real per-layer broadcast sizes:
  - indexer-K  ~0.25 - 8 MB / layer
  - dense KV    ~2 MB / layer (decode) ... 10 - 64 MB / layer (128k / large batch)

  (a) NCCL path : dist.broadcast(buf, src=owner, group=cp_subgroup)
  (b) IPC  path : owner cudaMemcpyPeerAsync -> each peer (dst.copy_(src))
  (c) bootstrap cost: TCPStore rendezvous + init_process_group + new_group.

Run via (CP=2, the primary deploy)::
    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1 LS_BENCH_WORLD=2 \
        mpirun --allow-run-as-root -n 2 python3 broadcast_primitive_bench.py
And CP=4::
    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3 LS_BENCH_WORLD=4 \
        mpirun --allow-run-as-root -n 4 python3 broadcast_primitive_bench.py

The IPC P2P baseline is the owner doing one cross-device copy PER peer (the
broadcast fan-out a custom IPC primitive would issue). P2P access is enabled
explicitly so the copy rides NVLink, not staged through host. We time the OWNER
rank's wall cost (it is the critical path: it must reach every peer).
"""
import os
import sys
import time

import torch
import torch.distributed as dist

ITERS = 200
WARMUP = 30
# Size sweep (bytes). Brackets indexer-K (0.25-8 MB) and dense KV (2-64 MB),
# plus a tiny 16 KB point to expose pure launch overhead.
SIZES_MB = [0.016, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def _log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def _bench_nccl(buf, src_rank, group, owner_is_self):
    """Median microseconds for one dist.broadcast on the CP subgroup,
    timed on the OWNER's critical path. Every rank participates; we report
    the owner's wall time (the cost that gates the per-layer compute)."""
    # Warmup
    for _ in range(WARMUP):
        dist.broadcast(buf, src=src_rank, group=group, async_op=False)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.broadcast(buf, src=src_rank, group=group, async_op=False)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    dist.barrier(group=group)
    samples.sort()
    return samples[len(samples) // 2]


def _bench_ipc_p2p(send_buf, peer_bufs, owner_is_self, peer_devices, stream):
    """Median microseconds for the OWNER to fan the payload out to every peer
    via a P2P (NVLink) cross-device copy -- the work a custom IPC broadcast
    primitive issues. Only the owner does copies; peers just wait. We time the
    owner's wall cost (issue all peer copies on one stream, then sync).

    Non-owner ranks return None (they have no fan-out work; the broadcast
    semantics are owner-push)."""
    if not owner_is_self:
        return None
    # Warmup
    for _ in range(WARMUP):
        with torch.cuda.stream(stream):
            for dst in peer_bufs:
                dst.copy_(send_buf, non_blocking=True)
        stream.synchronize()

    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            for dst in peer_bufs:
                dst.copy_(send_buf, non_blocking=True)
        stream.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
        ensure_cp_process_group)

    rank = mpi_rank()
    world = mpi_world_size()
    want_world = int(os.environ.get("LS_BENCH_WORLD", world))
    if world != want_world:
        _log(rank, f"FAIL world mismatch {world} != {want_world}")
        return 1

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    # Build a Mapping whose CP subgroup spans the WHOLE world (cp_size=world,
    # tp_size=1) so the carved subgroup == [0..world). This makes the NCCL
    # broadcast a CP=`world` broadcast (owner -> world-1 peers), matching the
    # CP=2 / CP=4 deploy fan-out exactly.
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=world,
        tp_size=1,
        cp_size=world,
        cp_config={"cp_type": "LAYERSPLIT"},
    )

    # ----- (c) one-time bootstrap cost -----
    t0 = time.perf_counter()
    cp_group, cp_group_ranks = ensure_cp_process_group(mapping)
    boot_ms = (time.perf_counter() - t0) * 1e3
    if cp_group is None or not dist.is_initialized():
        _log(rank, f"FAIL bootstrap cp_group={cp_group} "
                   f"dist_init={dist.is_initialized()}")
        return 1
    _log(rank, f"bootstrap_ms={boot_ms:.1f} cp_group_ranks={cp_group_ranks}")

    # Owner of the broadcast = global rank cp_group_ranks[0] (CP-local owner 0).
    owner_rank = cp_group_ranks[0]
    owner_is_self = (rank == owner_rank)

    # Enable P2P access from the owner to every peer for the IPC baseline.
    # (Owner pushes to peer device memory over NVLink.)
    peer_global_ranks = [r for r in cp_group_ranks if r != owner_rank]
    if owner_is_self:
        for pr in peer_global_ranks:
            # On a single node under one mpirun, global rank == device index.
            torch.cuda.set_device(owner_rank)  # ensure owner ctx is current
            can = torch.cuda.can_device_access_peer(owner_rank, pr)
            if can:
                # Establish the peer mapping (idempotent). torch enables P2P
                # lazily on first cross-device copy; this is a sanity probe.
                pass
        torch.cuda.set_device(rank)
        _log(rank, "owner: P2P access probed for peers "
                   f"{peer_global_ranks} "
                   f"can_access={[bool(torch.cuda.can_device_access_peer(owner_rank, pr)) for pr in peer_global_ranks]}")

    ipc_stream = torch.cuda.Stream(device=dev)

    rows = []
    for size_mb in SIZES_MB:
        nbytes = int(size_mb * 1024 * 1024)
        # NCCL buffer lives on THIS rank's device.
        nccl_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        if owner_is_self:
            nccl_buf.fill_(0xAB)

        nccl_us = _bench_nccl(nccl_buf, owner_rank, cp_group, owner_is_self)

        # IPC P2P: owner holds the send buffer on its device; allocate one
        # destination buffer per peer ON THE PEER'S DEVICE (the owner pushes
        # into peer memory). Only the owner times / issues copies.
        ipc_us = None
        if owner_is_self:
            send_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            send_buf.fill_(0xCD)
            peer_bufs = []
            for pr in peer_global_ranks:
                pdev = torch.device(f"cuda:{pr}")
                peer_bufs.append(
                    torch.empty(nbytes, dtype=torch.uint8, device=pdev))
            ipc_us = _bench_ipc_p2p(send_buf, peer_bufs, owner_is_self,
                                    peer_global_ranks, ipc_stream)
            del send_buf, peer_bufs
        del nccl_buf
        torch.cuda.empty_cache()

        if owner_is_self:
            rows.append((size_mb, nccl_us, ipc_us))
            speedup = (nccl_us / ipc_us) if (ipc_us and ipc_us > 0) else float("nan")
            _log(rank,
                 f"size={size_mb:>7.3f}MB  nccl={nccl_us:8.2f}us  "
                 f"ipc_p2p={ipc_us:8.2f}us  nccl/ipc={speedup:5.2f}x")
        dist.barrier(group=cp_group)

    if owner_is_self:
        print(f"\n=== BROADCAST PRIMITIVE BENCH (CP={world}, owner=rank "
              f"{owner_rank} -> {len(peer_global_ranks)} peer(s)) ===",
              flush=True)
        print(f"bootstrap_one_time_ms={boot_ms:.1f}", flush=True)
        print(f"{'size_MB':>9} {'nccl_us':>10} {'ipc_p2p_us':>12} "
              f"{'nccl/ipc':>9} {'winner':>8}", flush=True)
        for size_mb, nccl_us, ipc_us in rows:
            speedup = (nccl_us / ipc_us) if (ipc_us and ipc_us > 0) else float("nan")
            winner = "ipc" if (ipc_us and nccl_us > ipc_us) else "nccl"
            print(f"{size_mb:>9.3f} {nccl_us:>10.2f} {ipc_us:>12.2f} "
                  f"{speedup:>9.2f} {winner:>8}", flush=True)
        print("=== END BENCH ===", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

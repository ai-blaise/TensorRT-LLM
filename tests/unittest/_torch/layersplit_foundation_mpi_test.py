"""Phase-1A FOUNDATION test: exercise the REAL MPI bootstrap of the CP
broadcast process group and the M5e active-block broadcast over the carved
CP subgroup.

Run via::

    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        mpirun --allow-run-as-root -n 4 python3 foundation_test.py

Why mpirun and not torch.distributed.run: the production deploy launches its
workers under OpenMPI (mpi4py.futures.server), and on that path
``torch.distributed`` is NEVER initialized -- the TP/EP collectives use the
C++ custom-allreduce / IPC-workspace path. ``torch.distributed.run`` would
pre-initialize a world PG and thereby SKIP the very bootstrap we must validate
(``_ensure_torch_distributed_under_mpi``). Launching under mpirun reproduces
the production rank topology exactly: dist is cold, only MPI is up.

PASS criteria (all 4 ranks must satisfy):
  1. ensure_cp_process_group(mapping) bootstraps NCCL via the MPI-TCPStore
     rendezvous (dist was NOT pre-initialized) and returns a real
     (cp_group, cp_group_ranks).
  2. The carved CP subgroup ranks are correct: (0, 1) for world ranks 0/1 and
     (2, 3) for world ranks 2/3 -- the production cp_groups=[[0,1],[2,3]] shape.
  3. The M5e active-block broadcast over the carved subgroup is BIT-EXACT:
     every rank's cache_slot[active_ids] equals the owner's stamped value,
     and inactive blocks are untouched.
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main() -> int:
    # mpi4py is initialized lazily by the first MPI call inside tensorrt_llm.
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    mpi_r = mpi_rank()
    mpi_ws = mpi_world_size()
    if mpi_ws != 4:
        _log(mpi_r, f"FAIL expected MPI world_size=4, got {mpi_ws}")
        return 1

    # Bind this rank to its GPU. CUDA_VISIBLE_DEVICES=0,1,2,3 maps local
    # device i -> physical GPU i; under one mpirun on one node, rank == device.
    torch.cuda.set_device(mpi_r)
    _log(mpi_r,
         f"start: cuda_dev={torch.cuda.current_device()} "
         f"dist_initialized_before_bootstrap={dist.is_initialized()}")

    # ------------------------------------------------------------------
    # 1. Build the REAL production-shape Mapping (MpiTopology).
    #    world=4, tp=2, cp=2, moe_ep=2, LAYERSPLIT -> cp_groups=[[0,1],[2,3]],
    #    cp_group_pg raises NotImplementedError. This is the exact prefill
    #    worker shape (TP2 x CP2).
    # ------------------------------------------------------------------
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=4,
        rank=mpi_r,
        gpus_per_node=4,
        tp_size=2,
        cp_size=2,
        moe_ep_size=2,
        cp_config={"cp_type": "LAYERSPLIT"},
    )
    _log(mpi_r,
         f"Mapping={type(mapping).__name__} cp_size={mapping.cp_size} "
         f"cp_rank={mapping.cp_rank} cp_groups={mapping.cp_groups} "
         f"cp_group={mapping.cp_group}")

    # Confirm the base-Mapping cp_group_pg really raises (the GAP this fix
    # closes); if it does NOT raise here we are not on the MPI path.
    cp_group_pg_raised = False
    try:
        _ = mapping.cp_group_pg
    except NotImplementedError:
        cp_group_pg_raised = True
    _log(mpi_r, f"cp_group_pg_raises_NotImplementedError={cp_group_pg_raised}")

    # ------------------------------------------------------------------
    # 2. THE BOOTSTRAP UNDER TEST. dist is cold under mpirun, so this MUST
    #    hit _ensure_torch_distributed_under_mpi (TCPStore rendezvous via
    #    mpi_broadcast + init_process_group + collective new_group carve).
    # ------------------------------------------------------------------
    from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
        LayerSplitRuntimeState, ensure_cp_process_group)

    import time
    t0 = time.perf_counter()
    cp_group, cp_group_ranks = ensure_cp_process_group(mapping)
    t1 = time.perf_counter()
    _log(mpi_r,
         f"ensure_cp_process_group -> cp_group={cp_group} "
         f"cp_group_ranks={cp_group_ranks} "
         f"dist_initialized_after={dist.is_initialized()} "
         f"bootstrap_ms={(t1 - t0) * 1e3:.1f}")

    errors = []
    if not dist.is_initialized():
        errors.append("torch.distributed was NOT initialized by the bootstrap")
    if cp_group is None:
        errors.append("ensure_cp_process_group returned cp_group=None "
                      "(bootstrap or carve failed)")
    # Expected carved subgroup for the production cp_groups=[[0,1],[2,3]]:
    expected_ranks = (0, 1) if mpi_r in (0, 1) else (2, 3)
    if cp_group_ranks != expected_ranks:
        errors.append(f"cp_group_ranks={cp_group_ranks}, "
                      f"expected {expected_ranks}")

    if errors:
        for e in errors:
            _log(mpi_r, f"FAIL {e}")
        # Still try to tear down cleanly.
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        _log(mpi_r, "FAIL (foundation bootstrap/carve)")
        return 1

    # ------------------------------------------------------------------
    # 3. Build runtime state, bind the carved CP group, run M5e.
    # ------------------------------------------------------------------
    cfg = SimpleNamespace(
        layersplit_enabled=True,
        layersplit_owner_assignment="contiguous",
        layersplit_transfer_backend="auto",
        layersplit_all_cp_ranks_transfer=True,
    )
    num_layers = 8
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=cfg,
        num_layers=num_layers,
        cp_size=2,
        cp_rank=mapping.cp_rank,
    )
    state.bind_cp_group(cp_group, cp_group_ranks)
    _log(mpi_r,
         f"runtime: cp_group_ranks={state.cp_group_ranks} "
         f"owner_map={state.ownership.owner_map}")

    # Sanity: broadcast_src_rank must translate the CP-local owner into the
    # correct GLOBAL src rank for this rank's subgroup.
    # contiguous cp=2, 8 layers: cp-local owner 0 owns layers 0..3, owner 1
    # owns 4..7. For subgroup [0,1]: global src = {0 for L0..3, 1 for L4..7}.
    # For subgroup [2,3]: global src = {2 for L0..3, 3 for L4..7}.
    base = cp_group_ranks[0]
    for L in range(num_layers):
        local_owner = state.ownership.owner_of(L)
        expect_src = base + local_owner
        got_src = state.broadcast_src_rank(L)
        if got_src != expect_src:
            errors.append(f"broadcast_src_rank({L})={got_src}, "
                          f"expected {expect_src}")
    if errors:
        for e in errors:
            _log(mpi_r, f"FAIL {e}")
        dist.barrier()
        dist.destroy_process_group()
        return 1

    # ------------------------------------------------------------------
    # 4. M5e active-block broadcast over the carved subgroup, bit-exact.
    #    Mirrors test_layersplit_multiproc_nccl._worker_m5e but rides the
    #    MPI-bootstrapped CP subgroup, not dist.group.WORLD.
    # ------------------------------------------------------------------
    num_blocks = 64
    block_bytes = 256
    active_block_ids = torch.tensor([3, 7, 17, 42],
                                    dtype=torch.int64,
                                    device="cuda")
    cp_local_rank = mapping.cp_rank  # 0 or 1 within the subgroup

    for layer_idx in range(num_layers):
        owner_local = state.ownership.owner_of(layer_idx)
        # Per-rank pool initialized with a rank-distinctive byte. Receivers'
        # ACTIVE blocks must end up holding the owner's stamped bytes; their
        # INACTIVE blocks must be untouched.
        cache = torch.full((num_blocks, block_bytes),
                           (cp_local_rank + 1) * 10,
                           dtype=torch.uint8,
                           device="cuda")
        if cp_local_rank == owner_local:
            for slot, blk_id in enumerate(active_block_ids.tolist()):
                cache[blk_id, :] = (layer_idx * 17 + slot * 3) % 256

        inactive_mask = torch.ones(num_blocks, dtype=torch.bool, device="cuda")
        inactive_mask[active_block_ids] = False

        issued = state.maybe_broadcast_active_blocks(
            layer_idx=layer_idx,
            cache_slot=cache,
            active_block_ids=active_block_ids,
            cp_group=cp_group,
        )
        torch.cuda.synchronize()
        if not issued:
            errors.append(f"layer {layer_idx}: maybe_broadcast_active_blocks "
                          f"returned False (broadcast did NOT issue)")
            break

        # Active blocks must now hold the OWNER's bytes on EVERY rank.
        for slot, blk_id in enumerate(active_block_ids.tolist()):
            expected = (layer_idx * 17 + slot * 3) % 256
            actual = cache[blk_id, 0].item()
            if actual != expected:
                errors.append(
                    f"layer {layer_idx} block {blk_id}: got {actual}, "
                    f"expected {expected} (owner_local={owner_local})")
                break
        # Inactive blocks must be untouched.
        inactive_value = (cp_local_rank + 1) * 10
        inactive_blocks = cache[inactive_mask]
        if not (inactive_blocks == inactive_value).all().item():
            diff = (inactive_blocks != inactive_value).any(dim=1).sum().item()
            errors.append(f"layer {layer_idx}: {diff} inactive blocks were "
                          f"modified (M5e must touch only active blocks)")

    # Barrier on the carved subgroup so a slow rank doesn't tear down the PG
    # while a peer is mid-collective.
    dist.barrier(group=cp_group)

    if errors:
        for e in errors:
            _log(mpi_r, f"FAIL {e}")
        dist.barrier()
        dist.destroy_process_group()
        _log(mpi_r, "FAIL (M5e broadcast)")
        return 1

    _log(mpi_r,
         f"PASS subgroup_ranks={cp_group_ranks} owner_map_ok "
         f"src_rank_ok m5e_bitexact layers={num_layers} "
         f"active_blocks={active_block_ids.tolist()}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

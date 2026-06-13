#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the persistent grid size the dense kernel uses for L-batch problems,
to understand whether L-batching amortizes ramp (1 persistent wave) or just
launches a bigger grid (no amortization). Computes num_ctas_mnl and the grid
the StaticPersistentTileScheduler would pick, for L=1 and L=4."""
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch


def probe(M, N, K, L, tile_mn=(128, 128), cluster_mn=(1, 1)):
    # Mirror _compute_grid: tile the (M,N,L) C output by the CTA tile.
    import math
    tile_m, tile_n = tile_mn
    ctas_m = math.ceil(M / tile_m)
    ctas_n = math.ceil(N / tile_n)
    num_ctas_mnl = ctas_m * ctas_n * L
    hw = cutlass.utils.HardwareInfo()
    max_active = hw.get_max_active_clusters(cluster_mn[0] * cluster_mn[1])
    sm_count = torch.cuda.get_device_properties().multi_processor_count
    return num_ctas_mnl, max_active, sm_count


def main():
    torch.cuda.set_device(0)
    sm = torch.cuda.get_device_properties().multi_processor_count
    print(f"# B200 SM count = {sm}")
    print(f"# {'shape':16s} {'tile':>10s} {'L':>2s} "
          f"{'tiles(ctas)':>12s} {'max_active_clusters':>20s} {'waves':>7s}")
    SHAPES = {"kv_a": (2112, 7168), "q_b": (24576, 1536),
              "o_proj": (7168, 16384), "shared_gu": (4096, 7168)}
    for name, (N, K) in SHAPES.items():
        for tile in [(128, 128), (128, 256)]:
            for L in [1, 4]:
                nctas, maxa, smc = probe(16, N, K, L, tile_mn=tile)
                waves = nctas / maxa if maxa else 0
                print(f"  {name:16s} {str(tile):>10s} {L:2d} "
                      f"{nctas:12d} {maxa:20d} {waves:7.2f}")


if __name__ == "__main__":
    main()

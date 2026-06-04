# SPDX-License-Identifier: Apache-2.0
"""Single-config robust validator for the WarpDecode fused megakernel.

One config per process (fresh CUDA context -> immune to IMA-poisoning from a
bad neighbor config). Interleaved mega-vs-seq timing cancels shared-GPU
contention. Reports median over many blocks + correctness cos.
"""
import argparse
import statistics
import torch
import torch.nn.functional as F

import warpdecode_mega_driver as md


class A:
    pass


def bench(launch, it, blocks, warm=40):
    for _ in range(warm):
        launch()
    torch.cuda.synchronize()
    ms = []
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            launch()
        e.record()
        torch.cuda.synchronize()
        ms.append(s.elapsed_time(e) / it * 1000.0)
    return statistics.median(ms), min(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-m", type=int, default=128)
    ap.add_argument("--fc1-n", type=int, default=256)
    ap.add_argument("--fc2-n", type=int, default=160)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--blocks", type=int, default=8)
    p = ap.parse_args()

    a = A()
    a.hidden = 7168; a.inter = 2048; a.hot = 6; a.ntok = 8
    a.tile_m = p.tile_m; a.tile_n = p.fc1_n; a.tile_n_fc2 = p.fc2_n
    a.top_k = 8; a.vec_f32 = True; a.iters = p.iters; a.blocks = p.blocks

    mega, (out_gpu, _, _, seq, _) = md.build_mega(a)

    # correctness (zero between, FC2 finalize is atomic-add)
    out_gpu.zero_(); seq(); torch.cuda.synchronize(); seq_out = out_gpu.clone()
    out_gpu.zero_(); mega(); torch.cuda.synchronize(); mega_out = out_gpu.clone()
    cos = F.cosine_similarity(mega_out.float().flatten(), seq_out.float().flatten(), dim=0).item()

    # interleaved A/B/A/B timing
    mm1, mn1 = bench(mega, p.iters, p.blocks)
    ss1, sn1 = bench(seq, p.iters, p.blocks)
    mm2, mn2 = bench(mega, p.iters, p.blocks)
    ss2, sn2 = bench(seq, p.iters, p.blocks)
    mega_med = statistics.median([mm1, mm2])
    seq_med = statistics.median([ss1, ss2])
    mega_min = min(mn1, mn2)

    base = 44.18
    print(f"VALID tile_m={p.tile_m} fc1_n={p.fc1_n} fc2_n={p.fc2_n} "
          f"mega_med={mega_med:.2f} mega_min={mega_min:.2f} seq_med={seq_med:.2f} "
          f"delta_fuse={mega_med - seq_med:+.2f} vs_prod={mega_med - base:+.2f} "
          f"({'BEATS' if mega_med < base else 'SLOWER'}) cos={cos:.5f}", flush=True)


if __name__ == "__main__":
    main()

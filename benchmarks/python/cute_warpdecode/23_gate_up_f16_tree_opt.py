"""WarpDecode Round 4: on top of R3 (F16 accum, 3.75 TB/s), collapse the within-block reduction
dependency depth. Baseline sums 16 elements serially (depth 16). R4 sums the two halves' 8-wide
value-product vectors element-wise (1 SIMD F16 vector add) then tree-reduces 8->1 (depth 3) = depth
~4 vs 16 -> more ILP. Scale-once-per-block in F32 (preserve precision), F32 across blocks. CZS = cosine."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
WARPS = 8

@cute.kernel
def gu_k(mW, mX, mSFW, mSFX, mEids, mInter, NB: cutlass.Constexpr, INTER: cutlass.Constexpr):
    bidx, _, bidz = cute.arch.block_idx(); tidx, _, _ = cute.arch.thread_idx(); warp = tidx // 32; lane = tidx % 32
    j = bidx * WARPS + warp; t = bidz // 8; e = mEids[bidz]
    gacc = cutlass.Float32(0.0); uacc = cutlass.Float32(0.0); blk = lane
    while blk < NB:
        sfx = cutlass.Float32(mSFX[t, blk]); sfg = cutlass.Float32(mSFW[e, j, blk]) * sfx; sfu = cutlass.Float32(mSFW[e, j + INTER, blk]) * sfx
        w0 = blk * 2; w1 = blk * 2 + 1
        xw0 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t, w0].ir_value()), (8,), cutlass.Float16)
        xw1 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t, w1].ir_value()), (8,), cutlass.Float16)
        gw0 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j, w0].ir_value()), (8,), cutlass.Float16)
        gw1 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j, w1].ir_value()), (8,), cutlass.Float16)
        uw0 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j + INTER, w0].ir_value()), (8,), cutlass.Float16)
        uw1 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j + INTER, w1].ir_value()), (8,), cutlass.Float16)
        gv = gw0 * xw0 + gw1 * xw1
        uv = uw0 * xw0 + uw1 * xw1
        g01 = gv[0] + gv[1]; g23 = gv[2] + gv[3]; g45 = gv[4] + gv[5]; g67 = gv[6] + gv[7]
        u01 = uv[0] + uv[1]; u23 = uv[2] + uv[3]; u45 = uv[4] + uv[5]; u67 = uv[6] + uv[7]
        gsh = (g01 + g23) + (g45 + g67); ush = (u01 + u23) + (u45 + u67)
        gacc = gacc + gsh.to(cutlass.Float32) * sfg; uacc = uacc + ush.to(cutlass.Float32) * sfu
        blk = blk + 32
    gg = cute.arch.warp_reduction_sum(gacc); uu = cute.arch.warp_reduction_sum(uacc)
    if lane == 0:
        silu = gg * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.arch.exp(-gg))); mInter[bidz, j] = (silu * uu).to(cutlass.BFloat16)

@cute.jit
def launch(mW, mX, mSFW_raw, mSFX_raw, mEids, mInter, E: cutlass.Constexpr, M2: cutlass.Constexpr, H: cutlass.Constexpr, T: cutlass.Constexpr, LP: cutlass.Constexpr, INTER: cutlass.Constexpr, stream):
    NB = H // 16
    mSFW = cute.make_tensor(mSFW_raw.iterator, cute.make_layout((E, M2, NB), stride=(M2 * NB, NB, 1)))
    mSFX = cute.make_tensor(mSFX_raw.iterator, cute.make_layout((T, NB), stride=(NB, 1)))
    gu_k(mW, mX, mSFW, mSFX, mEids, mInter, NB, INTER).launch(grid=(cute.ceil_div(INTER, WARPS), 1, LP), block=(WARPS * 32, 1, 1), stream=stream)

dev = "cuda"; torch.manual_seed(0)
E, INTER, H, T, TOPK = 16, 2048, 7168, 4, 8; M2 = 2 * INTER; LP = T * TOPK; NWh = H // 8; NB = H // 16
Wu = torch.randint(0, 2**31, (E, M2, NWh), device=dev, dtype=torch.int32); SFW = torch.randint(1, 15, (E, M2, NB), device=dev, dtype=torch.uint8).contiguous()
Xu = torch.randint(0, 2**31, (T, NWh), device=dev, dtype=torch.int32); SFX = torch.randint(1, 15, (T, NB), device=dev, dtype=torch.uint8).contiguous()
eids = torch.randint(0, E, (LP,), device=dev, dtype=torch.int32)
Inter = torch.zeros(LP, INTER, device=dev, dtype=torch.bfloat16)
E2 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]; lut = torch.tensor(E2, device=dev)
def deq(p, s, KK):
    R = p.shape[0]; pu = p.view(torch.uint8).view(R, KK // 8, 4); v = torch.empty(R, KK, device=dev)
    for b in range(4):
        lo = (pu[:, :, b] & 0xF).long(); hi = ((pu[:, :, b] >> 4) & 0xF).long(); v[:, (b * 2)::8] = lut[lo]; v[:, (b * 2 + 1)::8] = lut[hi]
    return v * s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
Wf = deq(Wu.reshape(E * M2, NWh), SFW.reshape(E * M2, NB), H).reshape(E, M2, H); Xf = deq(Xu, SFX, H)
ref = torch.empty(LP, INTER, device=dev)
for l in range(LP):
    t = l // TOPK; e = eids[l]; g = Wf[e, :INTER] @ Xf[t]; u = Wf[e, INTER:] @ Xf[t]; ref[l] = ((g * torch.sigmoid(g)) * u).to(torch.bfloat16)
mW = from_dlpack(Wu); mX = from_dlpack(Xu); mSFW = from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX = from_dlpack(SFX.view(torch.float8_e4m3fn)); mEids = from_dlpack(eids); mInter = from_dlpack(Inter)
cur = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp = cute.compile(launch, mW, mX, mSFW, mSFX, mEids, mInter, E, M2, H, T, LP, INTER, cur)
comp(mW, mX, mSFW, mSFX, mEids, mInter, cur); torch.cuda.synchronize()
cos = torch.nn.functional.cosine_similarity(Inter.float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
def bench(it=80, wu=20):
    f = lambda: comp(mW, mX, mSFW, mSFX, mEids, mInter, cur)
    for _ in range(wu): f()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1000
t_us = bench(); wb = LP * (M2 * H // 2)
print(f"R4 tree-reduce+F16 gate_up: cos={cos:.5f} {t_us:.1f}us {wb/(t_us*1e-6)/1e12:.2f}TB/s (R3=3.75, baseline=3.14, Cursor=3.95)", flush=True)

# robustness sweep: vary token count incl Cursor's B=32, multiple reps
print("--- robustness sweep (B=token count; min over 4 reps) ---", flush=True)
for Tn in [4, 16, 32]:
    LPn = Tn * TOPK; NWh2 = H // 8
    Xu2 = torch.randint(0, 2**31, (Tn, NWh2), device=dev, dtype=torch.int32); SFX2 = torch.randint(1, 15, (Tn, NB), device=dev, dtype=torch.uint8).contiguous()
    eids2 = torch.randint(0, E, (LPn,), device=dev, dtype=torch.int32); Inter2 = torch.zeros(LPn, INTER, device=dev, dtype=torch.bfloat16)
    mX2 = from_dlpack(Xu2); mSFX2 = from_dlpack(SFX2.view(torch.float8_e4m3fn)); mEids2 = from_dlpack(eids2); mInter2 = from_dlpack(Inter2)
    comp2 = cute.compile(launch, mW, mX2, mSFW, mSFX2, mEids2, mInter2, E, M2, H, Tn, LPn, INTER, cur)
    comp2(mW, mX2, mSFW, mSFX2, mEids2, mInter2, cur); torch.cuda.synchronize()
    Xf2 = deq(Xu2, SFX2, H); ref2 = torch.empty(LPn, INTER, device=dev)
    for l in range(LPn):
        tt = l // TOPK; ee = eids2[l]; gg2 = Wf[ee, :INTER] @ Xf2[tt]; uu2 = Wf[ee, INTER:] @ Xf2[tt]; ref2[l] = ((gg2 * torch.sigmoid(gg2)) * uu2).to(torch.bfloat16)
    cos2 = torch.nn.functional.cosine_similarity(Inter2.float().flatten().unsqueeze(0), ref2.float().flatten().unsqueeze(0)).item()
    best = 1e9
    for _ in range(4):
        f = lambda: comp2(mW, mX2, mSFW, mSFX2, mEids2, mInter2, cur)
        for _ in range(15): f()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(80): f()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / 80 * 1000)
    wb2 = LPn * (M2 * H // 2)
    print(f"  B={Tn:>2} (LP={LPn:>3}): cos={cos2:.5f} {best:.1f}us {wb2/(best*1e-6)/1e12:.2f}TB/s = {wb2/(best*1e-6)/1e12/6.8*100:.0f}% of 6.8 peak", flush=True)

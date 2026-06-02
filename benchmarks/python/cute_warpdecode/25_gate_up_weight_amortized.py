"""WarpDecode Round 8 (final): weight-amortization. At B=32 with 16 local experts each expert serves
~16 pairs, so R4 re-loads+re-dequantizes each expert weight ~16x (memory-bound). R8 realizes the
grouped-GEMM amortization principle (Veitner grouped post) at thread level: warp owns (expert,neuron),
processes 2 tokens/pass, loads+cvt the weight ONCE per block and reuses for both tokens (x is per-token).
Halves weight load+cvt per token. R4's F16+tree kept. CZS = cosine vs FP32 ref for BOTH tokens."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
WARPS = 8

@cute.kernel
def gu_k(mW, mX, mSFW, mSFX, mGE, mGT0, mGT1, mGV, mGP0, mGP1, mInter, NB: cutlass.Constexpr, INTER: cutlass.Constexpr):
    bidx, _, bidz = cute.arch.block_idx(); tidx, _, _ = cute.arch.thread_idx(); warp = tidx // 32; lane = tidx % 32
    j = bidx * WARPS + warp; g = bidz; e = mGE[g]; t0 = mGT0[g]; t1 = mGT1[g]; v1 = mGV[g]
    g0 = cutlass.Float32(0.0); u0 = cutlass.Float32(0.0); g1 = cutlass.Float32(0.0); u1 = cutlass.Float32(0.0)
    blk = lane
    while blk < NB:
        sfw_g = cutlass.Float32(mSFW[e, j, blk]); sfw_u = cutlass.Float32(mSFW[e, j + INTER, blk])
        sfx0 = cutlass.Float32(mSFX[t0, blk]); w0 = blk * 2; w1 = blk * 2 + 1
        gw0 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j, w0].ir_value()), (8,), cutlass.Float16)
        gw1 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j, w1].ir_value()), (8,), cutlass.Float16)
        uw0 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j + INTER, w0].ir_value()), (8,), cutlass.Float16)
        uw1 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e, j + INTER, w1].ir_value()), (8,), cutlass.Float16)
        x00 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t0, w0].ir_value()), (8,), cutlass.Float16)
        x01 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t0, w1].ir_value()), (8,), cutlass.Float16)
        gv = gw0 * x00 + gw1 * x01; uv = uw0 * x00 + uw1 * x01
        gsh = (gv[0] + gv[1] + gv[2] + gv[3]) + (gv[4] + gv[5] + gv[6] + gv[7])
        ush = (uv[0] + uv[1] + uv[2] + uv[3]) + (uv[4] + uv[5] + uv[6] + uv[7])
        g0 = g0 + gsh.to(cutlass.Float32) * sfw_g * sfx0; u0 = u0 + ush.to(cutlass.Float32) * sfw_u * sfx0
        if v1 > 0:
            sfx1 = cutlass.Float32(mSFX[t1, blk])
            y00 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t1, w0].ir_value()), (8,), cutlass.Float16)
            y01 = TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t1, w1].ir_value()), (8,), cutlass.Float16)
            gv1 = gw0 * y00 + gw1 * y01; uv1 = uw0 * y00 + uw1 * y01
            gs1 = (gv1[0] + gv1[1] + gv1[2] + gv1[3]) + (gv1[4] + gv1[5] + gv1[6] + gv1[7])
            us1 = (uv1[0] + uv1[1] + uv1[2] + uv1[3]) + (uv1[4] + uv1[5] + uv1[6] + uv1[7])
            g1 = g1 + gs1.to(cutlass.Float32) * sfw_g * sfx1; u1 = u1 + us1.to(cutlass.Float32) * sfw_u * sfx1
        blk = blk + 32
    gg0 = cute.arch.warp_reduction_sum(g0); uu0 = cute.arch.warp_reduction_sum(u0)
    gg1 = cute.arch.warp_reduction_sum(g1); uu1 = cute.arch.warp_reduction_sum(u1)
    if lane == 0:
        s0 = gg0 * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.arch.exp(-gg0))); mInter[mGP0[g], j] = (s0 * uu0).to(cutlass.BFloat16)
        if v1 > 0:
            s1 = gg1 * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.arch.exp(-gg1))); mInter[mGP1[g], j] = (s1 * uu1).to(cutlass.BFloat16)

@cute.jit
def launch(mW, mX, mSFW_raw, mSFX_raw, mGE, mGT0, mGT1, mGV, mGP0, mGP1, mInter, E: cutlass.Constexpr, M2: cutlass.Constexpr, H: cutlass.Constexpr, T: cutlass.Constexpr, NG: cutlass.Constexpr, INTER: cutlass.Constexpr, stream):
    NB = H // 16
    mSFW = cute.make_tensor(mSFW_raw.iterator, cute.make_layout((E, M2, NB), stride=(M2 * NB, NB, 1)))
    mSFX = cute.make_tensor(mSFX_raw.iterator, cute.make_layout((T, NB), stride=(NB, 1)))
    gu_k(mW, mX, mSFW, mSFX, mGE, mGT0, mGT1, mGV, mGP0, mGP1, mInter, NB, INTER).launch(grid=(cute.ceil_div(INTER, WARPS), 1, NG), block=(WARPS * 32, 1, 1), stream=stream)

dev = "cuda"; torch.manual_seed(0)
E, INTER, H, T, TOPK = 16, 2048, 7168, 32, 8; M2 = 2 * INTER; LP = T * TOPK; NWh = H // 8; NB = H // 16
Wu = torch.randint(0, 2**31, (E, M2, NWh), device=dev, dtype=torch.int32); SFW = torch.randint(1, 15, (E, M2, NB), device=dev, dtype=torch.uint8).contiguous()
Xu = torch.randint(0, 2**31, (T, NWh), device=dev, dtype=torch.int32); SFX = torch.randint(1, 15, (T, NB), device=dev, dtype=torch.uint8).contiguous()
eids = torch.randint(0, E, (LP,), device=dev, dtype=torch.int32)
# group pairs by expert, chunk into pairs-of-2 (token = pair//TOPK)
order = torch.argsort(eids); ge, gt0, gt1, gv, gp0, gp1 = [], [], [], [], [], []
i = 0; ordl = order.tolist(); eidl = eids.tolist()
while i < LP:
    p0 = ordl[i]; e0 = eidl[p0]
    if i + 1 < LP and eidl[ordl[i + 1]] == e0:
        p1 = ordl[i + 1]; ge.append(e0); gt0.append(p0 // TOPK); gt1.append(p1 // TOPK); gv.append(1); gp0.append(p0); gp1.append(p1); i += 2
    else:
        ge.append(e0); gt0.append(p0 // TOPK); gt1.append(0); gv.append(0); gp0.append(p0); gp1.append(0); i += 1
NG = len(ge)
mGE = from_dlpack(torch.tensor(ge, device=dev, dtype=torch.int32)); mGT0 = from_dlpack(torch.tensor(gt0, device=dev, dtype=torch.int32)); mGT1 = from_dlpack(torch.tensor(gt1, device=dev, dtype=torch.int32))
mGV = from_dlpack(torch.tensor(gv, device=dev, dtype=torch.int32)); mGP0 = from_dlpack(torch.tensor(gp0, device=dev, dtype=torch.int32)); mGP1 = from_dlpack(torch.tensor(gp1, device=dev, dtype=torch.int32))
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
    t = l // TOPK; e = eids[l]; gg = Wf[e, :INTER] @ Xf[t]; uu = Wf[e, INTER:] @ Xf[t]; ref[l] = ((gg * torch.sigmoid(gg)) * uu).to(torch.bfloat16)
mW = from_dlpack(Wu); mX = from_dlpack(Xu); mSFW = from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX = from_dlpack(SFX.view(torch.float8_e4m3fn)); mInter = from_dlpack(Inter)
cur = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp = cute.compile(launch, mW, mX, mSFW, mSFX, mGE, mGT0, mGT1, mGV, mGP0, mGP1, mInter, E, M2, H, T, NG, INTER, cur)
comp(mW, mX, mSFW, mSFX, mGE, mGT0, mGT1, mGV, mGP0, mGP1, mInter, cur); torch.cuda.synchronize()
cos = torch.nn.functional.cosine_similarity(Inter.float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
def bench(it=80, wu=20):
    f = lambda: comp(mW, mX, mSFW, mSFX, mGE, mGT0, mGT1, mGV, mGP0, mGP1, mInter, cur)
    for _ in range(wu): f()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1000
t_us = bench(); wb = LP * (M2 * H // 2)
wb_amort = NG * (M2 * H // 2)  # weight actually loaded (once per group)
print(f"R8 weight-amortized gate_up (B={T}, NG={NG} groups, {LP/NG:.1f} tok/group): cos={cos:.5f} {t_us:.1f}us", flush=True)
print(f"   per-pair-equiv throughput = {wb/(t_us*1e-6)/1e12:.2f}TB/s (R4 per-pair=4.52); R4 latency at B32 was ~832us -> speedup {832/t_us:.2f}x", flush=True)

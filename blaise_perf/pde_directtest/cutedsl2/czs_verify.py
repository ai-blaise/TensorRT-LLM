"""R2 Phase-3 CZS verification with the REAL czs_py pybind (built this round).
(A) Try czs.frontends.cutedsl.verify on a traced NVFP4 blockscaled GEMM (real compiled layouts).
(B) Hand-encode the WIN tactic's layouts/TMA/MMA as a czs.Module and run_all_passes (now via REAL pybind,
    not a stub) -> structural legality verdicts for the deliverable config.
Run inside the image (needs cutlass.cute) with CZS on PYTHONPATH."""
import sys, importlib
sys.path.insert(0, "/czs_py")  # mounted CZS/python with the built _native.so
import czs
print("CZS", czs.__version__, "via REAL pybind:", hasattr(czs._native, "__version__"), flush=True)

# (A) real-kernel trace verify
try:
    from czs.frontends import cutedsl as fe
    print("cutedsl frontend available (cutlass.cute importable):", fe.is_available(), flush=True)
except Exception as e:
    print("frontend import:", type(e).__name__, str(e)[:80], flush=True)

# (B) Hand-encode the WIN config layouts -> Module -> run_all_passes via REAL pybind.
# WIN: nvf4 sf_vec=16, mma_tiler (256,64), cluster (4,1), swap_ab (N=7168 on M-axis), 2cta.
# Per-CTA MMA atom M128 N64 K64 (256/2 with 2cta). FP4 16U4 TMA: 32B base, K-leading mult128, swz128B.
def L(shape, stride, label):
    l = czs.Layout(); l.shape=[czs.IntDim.Static(s) for s in shape]
    l.stride=[czs.IntDim.Static(s) for s in stride]; l.label=label; return l
mod = czs.Module()
# A-operand SMEM tile (per-CTA): M128 x K64 fp4, K-leading, 128B-swizzle stand-in
mod.append_layout(L([128,64],[64,1],"A_smem_M128K64_Kmajor"))
# B-operand SMEM tile: N64 x K64 fp4 (swap_ab puts activation N on the kernel-N)
mod.append_layout(L([64,64],[64,1],"B_smem_N64K64_Kmajor"))
# Accumulator TMEM: M128 x N64 fp32 (lanes x cols)
mod.append_layout(L([128,64],[64,1],"Acc_tmem_M128N64"))
# SFA TMEM (nvf4 block16): 16 cols region; SFB TMEM: 32 cols region (4x mxf4)
mod.append_layout(L([128,16],[16,1],"SFA_tmem_nvf4_16col"))
mod.append_layout(L([64,32],[32,1],"SFB_tmem_nvf4_32col"))
# Output bf16 STG.128 tile
mod.append_layout(L([128,64],[64,1],"C_out_bf16_M128N64"))
ctx = czs.Context()
verdicts = czs.run_all_passes(mod, ctx)
P=sum(1 for v in verdicts if v.result.proved())
D=sum(1 for v in verdicts if v.result.disproved())
U=sum(1 for v in verdicts if v.result.unknown())
print(f"run_all_passes (REAL pybind): {P} Proved | {D} Disproved | {U} Unknown  (n={len(verdicts)})", flush=True)
for v in verdicts:
    r=v.result; o="Proved" if r.proved() else ("Disproved" if r.disproved() else "Unknown")
    print(f"   {o:9} {getattr(v,'label',getattr(r,'label',''))}", flush=True)
# also: layout-legality pass alone on the 6 layouts
try:
    ll = czs.run_layout_legality(mod, ctx)
    LP=sum(1 for v in ll if v.result.proved())
    print(f"run_layout_legality: {LP}/{len(ll)} layouts injective/legal", flush=True)
except Exception as e:
    print("layout_legality:", type(e).__name__, str(e)[:60], flush=True)
print("DONE", flush=True)

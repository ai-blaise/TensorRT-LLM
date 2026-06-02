"""Full matrix: OPTIMIZED faithful WarpDecode (multi-output/warp) vs native NVFP4. Per-rank decode."""
import os, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
os.environ.setdefault("TRTLLM_ENABLE_PDL","1")
WARPS=8; NPW=1; HPW=8; H,INTER,TOPK,NE=7168,2048,8,128; M2=2*INTER
dev="cuda"; torch.manual_seed(0)

@cute.kernel
def gu_k(mW,mX,mSFW,mSFX,mEids,mInter,NWk:cutlass.Constexpr,INTER:cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx(); warp=tidx//32; lane=tidx%32
    j0=(bidx*WARPS+warp)*NPW; t=bidz//TOPK; e=mEids[bidz]
    g=cute.make_fragment(NPW,cutlass.Float32); u=cute.make_fragment(NPW,cutlass.Float32)
    for q in cutlass.range_constexpr(NPW): g[q]=cutlass.Float32(0.0); u[q]=cutlass.Float32(0.0)
    w=lane
    while w<NWk:
        xw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t,w].ir_value()),(8,),cutlass.Float16); sfx=cutlass.Float32(mSFX[t,w//2])
        for q in cutlass.range_constexpr(NPW):
            j=j0+q
            gw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e,j,w].ir_value()),(8,),cutlass.Float16)
            uw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e,j+INTER,w].ir_value()),(8,),cutlass.Float16)
            gp=(gw*xw).to(cutlass.Float32); upp=(uw*xw).to(cutlass.Float32); gs=cutlass.Float32(0.0); us=cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(8): gs=gs+gp[i]; us=us+upp[i]
            g[q]=g[q]+gs*cutlass.Float32(mSFW[e,j,w//2])*sfx; u[q]=u[q]+us*cutlass.Float32(mSFW[e,j+INTER,w//2])*sfx
        w=w+32
    for q in cutlass.range_constexpr(NPW):
        gg=cute.arch.warp_reduction_sum(g[q]); uu=cute.arch.warp_reduction_sum(u[q])
        if lane==0:
            silu=gg*(cutlass.Float32(1.0)/(cutlass.Float32(1.0)+cute.arch.exp(-gg))); mInter[bidz,j0+q]=(silu*uu).to(cutlass.BFloat16)
@cute.jit
def launch_gu(mW,mX,mSFW_raw,mSFX_raw,mEids,mInter,E:cutlass.Constexpr,M2:cutlass.Constexpr,H:cutlass.Constexpr,T:cutlass.Constexpr,LP:cutlass.Constexpr,INTER:cutlass.Constexpr,stream):
    NWk=H//8; sfk=H//16
    mSFW=cute.make_tensor(mSFW_raw.iterator,cute.make_layout((E,M2,sfk),stride=(M2*sfk,sfk,1)))
    mSFX=cute.make_tensor(mSFX_raw.iterator,cute.make_layout((T,sfk),stride=(sfk,1)))
    gu_k(mW,mX,mSFW,mSFX,mEids,mInter,NWk,INTER).launch(grid=(cute.ceil_div(INTER,WARPS*NPW),1,LP),block=(WARPS*32,1,1),stream=stream)

@cute.kernel
def dn_k(mW2,mI3,mSFW2,mEids,mRW,mOut,NWk:cutlass.Constexpr,TOPK:cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx(); warp=tidx//32; lane=tidx%32
    h0=(bidx*WARPS+warp)*HPW; t=bidz
    accs=cute.make_fragment(HPW,cutlass.Float32)
    for q in cutlass.range_constexpr(HPW): accs[q]=cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(TOPK):
        pair=t*TOPK+ks; e=mEids[pair]; rw=cutlass.Float32(mRW[pair]); w=lane
        parts=cute.make_fragment(HPW,cutlass.Float32)
        for q in cutlass.range_constexpr(HPW): parts[q]=cutlass.Float32(0.0)
        while w<NWk:
            iv=mI3[pair,w,None].load().to(cutlass.Float16)
            for q in cutlass.range_constexpr(HPW):
                dw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW2[e,h0+q,w].ir_value()),(8,),cutlass.Float16)
                prod=(dw*iv).to(cutlass.Float32); s=cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(8): s=s+prod[i]
                parts[q]=parts[q]+s*cutlass.Float32(mSFW2[e,h0+q,w//2])
            w=w+32
        for q in cutlass.range_constexpr(HPW): accs[q]=accs[q]+parts[q]*rw
    for q in cutlass.range_constexpr(HPW):
        o=cute.arch.warp_reduction_sum(accs[q])
        if lane==0: mOut[t,h0+q]=o.to(cutlass.BFloat16)
@cute.jit
def launch_dn(mW2,mI3,mSFW2_raw,mEids,mRW,mOut,E:cutlass.Constexpr,H:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,LP:cutlass.Constexpr,stream):
    NWk=INTER//8; sfk=INTER//16
    mSFW2=cute.make_tensor(mSFW2_raw.iterator,cute.make_layout((E,H,sfk),stride=(H*sfk,sfk,1)))
    dn_k(mW2,mI3,mSFW2,mEids,mRW,mOut,NWk,TOPK).launch(grid=(cute.ceil_div(H,WARPS*HPW),1,T),block=(WARPS*32,1,1),stream=stream)

cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def bench(fn,it=40,wu=10):
    for _ in range(wu): fn(cur)
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn(cur)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000

NAT={(8,16):81.6,(8,32):78.5,(4,16):151.1,(4,32):145.7,(2,16):246.9,(2,32):280.6}
print(f"{'cell':>10} {'LE':>3} {'LP':>4} {'WD_gu':>7} {'WD_dn':>7} {'WD_tot':>7} {'native':>7} {'speedup':>8}")
for (G,conc) in [(8,16),(8,32),(4,16),(4,32),(2,16),(2,32)]:
    LE=NE//G; LP=max(conc*TOPK//G,1); T=max(conc//G,1); NWh=H//8; NWi=INTER//8
    Wu=torch.randint(0,2**31,(LE,M2,NWh),device=dev,dtype=torch.int32); SFW=torch.randint(1,15,(LE,M2,H//16),device=dev,dtype=torch.uint8).contiguous()
    Xu=torch.randint(0,2**31,(T,NWh),device=dev,dtype=torch.int32); SFX=torch.randint(1,15,(T,H//16),device=dev,dtype=torch.uint8).contiguous()
    W2u=torch.randint(0,2**31,(LE,H,NWi),device=dev,dtype=torch.int32); SFW2=torch.randint(1,15,(LE,H,INTER//16),device=dev,dtype=torch.uint8).contiguous()
    eids=torch.randint(0,LE,(LP,),device=dev,dtype=torch.int32); rw=torch.rand(LP,device=dev,dtype=torch.float32)
    Inter=torch.zeros(LP,INTER,device=dev,dtype=torch.bfloat16); Inter3=Inter.view(LP,NWi,8); Out=torch.zeros(T,H,device=dev,dtype=torch.bfloat16)
    mW=from_dlpack(Wu); mX=from_dlpack(Xu); mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn)); mEids=from_dlpack(eids); mInter=from_dlpack(Inter)
    cgu=cute.compile(launch_gu,mW,mX,mSFW,mSFX,mEids,mInter,LE,M2,H,T,LP,INTER,cur)
    mW2=from_dlpack(W2u); mI3=from_dlpack(Inter3); mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mRW=from_dlpack(rw); mOut=from_dlpack(Out)
    cdn=cute.compile(launch_dn,mW2,mI3,mSFW2,mEids,mRW,mOut,LE,H,INTER,T,LP,cur)
    tg=bench(lambda st: cgu(mW,mX,mSFW,mSFX,mEids,mInter,st)); td=bench(lambda st: cdn(mW2,mI3,mSFW2,mEids,mRW,mOut,st))
    wd=tg+td; nt=NAT[(G,conc)]
    print(f"{('G'+str(G)+'c'+str(conc)):>10} {LE:>3} {LP:>4} {tg:7.1f} {td:7.1f} {wd:7.1f} {nt:7.1f} {nt/wd:7.2f}x")

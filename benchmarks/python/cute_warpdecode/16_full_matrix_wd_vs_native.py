"""Full-matrix faithful WarpDecode (Cursor) vs op-trt native NVFP4 baseline, per-rank decode work.
Both single-GPU local-compute comparison (the MoE layer is context-independent at decode).
WarpDecode = faithful gate_up + down (warp-per-output, shfl reduce, no atomic, BF16 intermediate).
Native = FP4BlockScaleMoERunner (op-trt grouped NVFP4 path)."""
import os, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
os.environ.setdefault("TRTLLM_ENABLE_PDL","1")
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner, ActType_TrtllmGen
WARPS=8; H,INTER,TOPK,NE,NG,TG,SV=7168,2048,8,128,8,4,16; M2=2*INTER
dev="cuda"; torch.manual_seed(0)

@cute.kernel
def gu_k(mW,mX,mSFW,mSFX,mEids,mInter,NW:cutlass.Constexpr,INTER:cutlass.Constexpr,TOPK:cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx(); warp=tidx//32; lane=tidx%32
    j=bidx*WARPS+warp; t=bidz//TOPK; e=mEids[bidz]
    gacc=cutlass.Float32(0.0); uacc=cutlass.Float32(0.0); w=lane
    while w<NW:
        xw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t,w].ir_value()),(8,),cutlass.Float16)
        gw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e,j,w].ir_value()),(8,),cutlass.Float16)
        uw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e,j+INTER,w].ir_value()),(8,),cutlass.Float16)
        sfx=cutlass.Float32(mSFX[t,w//2]); sfg=cutlass.Float32(mSFW[e,j,w//2]); sfu=cutlass.Float32(mSFW[e,j+INTER,w//2])
        gp=(gw*xw).to(cutlass.Float32); upp=(uw*xw).to(cutlass.Float32); gs=cutlass.Float32(0.0); us=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): gs=gs+gp[i]; us=us+upp[i]
        gacc=gacc+gs*sfg*sfx; uacc=uacc+us*sfu*sfx; w=w+32
    g=cute.arch.warp_reduction_sum(gacc); u=cute.arch.warp_reduction_sum(uacc)
    if lane==0:
        silu=g*(cutlass.Float32(1.0)/(cutlass.Float32(1.0)+cute.arch.exp(-g))); mInter[bidz,j]=(silu*u).to(cutlass.BFloat16)
@cute.jit
def launch_gu(mW,mX,mSFW_raw,mSFX_raw,mEids,mInter,E:cutlass.Constexpr,M2:cutlass.Constexpr,H:cutlass.Constexpr,T:cutlass.Constexpr,LP:cutlass.Constexpr,INTER:cutlass.Constexpr,TOPK:cutlass.Constexpr,stream):
    NW=H//8; sfk=H//16
    mSFW=cute.make_tensor(mSFW_raw.iterator,cute.make_layout((E,M2,sfk),stride=(M2*sfk,sfk,1)))
    mSFX=cute.make_tensor(mSFX_raw.iterator,cute.make_layout((T,sfk),stride=(sfk,1)))
    gu_k(mW,mX,mSFW,mSFX,mEids,mInter,NW,INTER,TOPK).launch(grid=(cute.ceil_div(INTER,WARPS),1,LP),block=(WARPS*32,1,1),stream=stream)

@cute.kernel
def dn_k(mW2,mInter,mSFW2,mEids,mRW,mOut,NW:cutlass.Constexpr,TOPK:cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx(); warp=tidx//32; lane=tidx%32
    h=bidx*WARPS+warp; t=bidz; acc=cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(TOPK):
        pair=t*TOPK+ks; e=mEids[pair]; rw=cutlass.Float32(mRW[pair]); w=lane; part=cutlass.Float32(0.0)
        while w<NW:
            dw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW2[e,h,w].ir_value()),(8,),cutlass.Float16); sfd=cutlass.Float32(mSFW2[e,h,w//2])
            s=cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(8): s=s+cutlass.Float32(dw[i])*cutlass.Float32(mInter[pair,w*8+i])
            part=part+s*sfd; w=w+32
        acc=acc+part*rw
    o=cute.arch.warp_reduction_sum(acc)
    if lane==0: mOut[t,h]=o.to(cutlass.BFloat16)
@cute.jit
def launch_dn(mW2,mInter,mSFW2_raw,mEids,mRW,mOut,E:cutlass.Constexpr,H:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,LP:cutlass.Constexpr,TOPK:cutlass.Constexpr,stream):
    NW=INTER//8; sfk=INTER//16
    mSFW2=cute.make_tensor(mSFW2_raw.iterator,cute.make_layout((E,H,sfk),stride=(H*sfk,sfk,1)))
    dn_k(mW2,mInter,mSFW2,mEids,mRW,mOut,NW,TOPK).launch(grid=(cute.ceil_div(H,WARPS),1,T),block=(WARPS*32,1,1),stream=stream)

cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def graph_bench(fn,it=50,wu=10):
    for _ in range(wu): fn(cur)
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn(cur)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000

# native runner cache by LE
def native_setup(LE):
    w13=torch.randint(0,256,(LE,M2,H//2),device=dev,dtype=torch.uint8); w13s=torch.randint(0,256,(LE,M2,H//SV),device=dev,dtype=torch.uint8).view(torch.float8_e4m3fn)
    w2=torch.randint(0,256,(LE,H,INTER//2),device=dev,dtype=torch.uint8); w2s=torch.randint(0,256,(LE,H,INTER//SV),device=dev,dtype=torch.uint8).view(torch.float8_e4m3fn)
    a1=torch.ones((LE,),device=dev,dtype=torch.float32)
    r=FP4BlockScaleMoERunner(NE,TOPK,NG,TG,INTER,0,LE,None,2,True,ActType_TrtllmGen.SwiGlu.value,tune_max_num_tokens=8192,use_dp=False)
    return r,w13,w13s,w2,w2s,a1

print("=== FULL MATRIX: faithful WarpDecode vs op-trt native NVFP4 (per-rank decode, ctx-independent) ===")
print(f"{'cell':>16} {'LE':>3} {'LP':>4} {'WD_gu':>8} {'WD_dn':>8} {'WD_tot':>8} {'native':>8} {'speedup':>8}")
for (G,conc) in [(8,16),(8,32),(4,16),(4,32),(2,16),(2,32)]:
    LE=NE//G; LP=max(conc*TOPK//G,1); T=max(conc//G,1)
    # faithful WD inputs (per-rank: LP pairs over LE local experts)
    Wu=torch.randint(0,2**31,(LE,M2,H//8),device=dev,dtype=torch.int32); SFW=torch.randint(1,15,(LE,M2,H//16),device=dev,dtype=torch.uint8).contiguous()
    Xu=torch.randint(0,2**31,(max(T,1),H//8),device=dev,dtype=torch.int32); SFX=torch.randint(1,15,(max(T,1),H//16),device=dev,dtype=torch.uint8).contiguous()
    W2u=torch.randint(0,2**31,(LE,H,INTER//8),device=dev,dtype=torch.int32); SFW2=torch.randint(1,15,(LE,H,INTER//16),device=dev,dtype=torch.uint8).contiguous()
    eids=torch.randint(0,LE,(LP,),device=dev,dtype=torch.int32); rw=torch.rand(LP,device=dev,dtype=torch.float32)
    Inter=torch.zeros(LP,INTER,device=dev,dtype=torch.bfloat16); Out=torch.zeros(max(T,1),H,device=dev,dtype=torch.bfloat16)
    LPp=LP; Tp=max(T,1)
    mW=from_dlpack(Wu); mX=from_dlpack(Xu); mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn)); mEids=from_dlpack(eids); mInter=from_dlpack(Inter)
    cgu=cute.compile(launch_gu,mW,mX,mSFW,mSFX,mEids,mInter,LE,M2,H,Tp,LPp,INTER,TOPK,cur)
    mW2=from_dlpack(W2u); mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mRW=from_dlpack(rw); mOut=from_dlpack(Out); mInter2=from_dlpack(Inter)
    cdn=cute.compile(launch_dn,mW2,mInter2,mSFW2,mEids,mRW,mOut,LE,H,INTER,Tp,LPp,TOPK,cur)
    t_gu=graph_bench(lambda st: cgu(mW,mX,mSFW,mSFX,mEids,mInter,st))
    t_dn=graph_bench(lambda st: cdn(mW2,mInter2,mSFW2,mEids,mRW,mOut,st))
    # native baseline (measured separately, graph+PDL, wd_native_baseline.py — runner not safely co-capturable here)
    NAT={(8,16):81.6,(8,32):78.5,(4,16):151.1,(4,32):145.7,(2,16):246.9,(2,32):280.6}
    t_nat=NAT[(G,conc)]
    wd=t_gu+t_dn
    print(f"{('G'+str(G)+' c'+str(conc)):>16} {LE:>3} {LP:>4} {t_gu:8.1f} {t_dn:8.1f} {wd:8.1f} {t_nat:8.1f} {t_nat/wd:8.2f}x")

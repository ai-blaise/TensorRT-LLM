import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
TM=8; TK=32

@cute.kernel
def gu_k(mW, mX, mSFW, mSFX, meidx, mtidx, mInter, UPOFF: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tk,tm,_=cute.arch.thread_idx()
    e=meidx[bidz]; t=mtidx[bidz]
    We=mW[e,None,None]; SFWe=mSFW[e,None,None]; Xt=mX[t,None]; SFXt=mSFX[t,None]
    gW=cute.local_tile(We,cute.slice_(MMA,(None,0,None)),(None,None))
    gSFW=cute.local_tile(SFWe,cute.slice_(MMA,(None,0,None)),(None,None))
    alloc=cutlass.utils.SmemAllocator()
    shg=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    shu=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    gacc=cutlass.Float32(0.0); uacc=cutlass.Float32(0.0)
    nw=cute.size(gW.layout[3].shape)
    for kt in cutlass.range(tk, nw, TK):
        bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(Xt[kt].ir_value()), (8,), cutlass.Float16)
        sfx=cutlass.Float32(SFXt[kt//2])
        gw=gW[tm,None,bidx,kt].load()
        gaf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(gw[0].ir_value()), (8,), cutlass.Float16)
        gp=(gaf*bf).to(cutlass.Float32); gs=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): gs=gs+gp[i]
        gacc=gacc+gs*cutlass.Float32(gSFW[tm,0,bidx,kt//2])*sfx
        uw=gW[tm,None,bidx+UPOFF,kt].load()
        uaf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(uw[0].ir_value()), (8,), cutlass.Float16)
        up=(uaf*bf).to(cutlass.Float32); us=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): us=us+up[i]
        uacc=uacc+us*cutlass.Float32(gSFW[tm,0,bidx+UPOFF,kt//2])*sfx
    shg[tm,tk]=gacc; shu[tm,tk]=uacc
    cute.arch.sync_threads()
    if tk==0:
        g=cutlass.Float32(0.0); u=cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(TK): g=g+shg[tm,r]; u=u+shu[tm,r]
        silu=g*(cutlass.Float32(1.0)/(cutlass.Float32(1.0)+cute.arch.exp(-g)))
        mInter[bidx*TM+tm, bidz]=silu*u

@cute.jit
def launch(mW,mX,mSFW_raw,mSFX_raw,meidx,mtidx,mInter, E:cutlass.Constexpr,M2:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr,INTER:cutlass.Constexpr, stream):
    MMA=(TM,1,1); sfk=HIDDEN//16
    mSFW=cute.make_tensor(mSFW_raw.iterator, cute.make_layout((E,M2,sfk),stride=(M2*sfk,sfk,1)))
    mSFX=cute.make_tensor(mSFX_raw.iterator, cute.make_layout((T,sfk),stride=(sfk,1)))
    gu_k(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,INTER//TM,MMA).launch(grid=(cute.ceil_div(INTER,TM),1,L),block=(TK,TM,1),stream=stream)

dev="cuda"; torch.manual_seed(2)
E,INTER,HIDDEN,T,L = 16,2048,7168,32,32; M2=2*INTER; nw=HIDDEN//8; sfk=HIDDEN//16
Wu=torch.randint(0,2**31,(E,M2,nw),device=dev,dtype=torch.int32); SFW=torch.randint(1,15,(E,M2,sfk),device=dev,dtype=torch.uint8).contiguous()
Xu=torch.randint(0,2**31,(T,nw),device=dev,dtype=torch.int32); SFX=torch.randint(1,15,(T,sfk),device=dev,dtype=torch.uint8).contiguous()
eidx=torch.randint(0,E,(L,),device=dev,dtype=torch.int32); tidx=torch.randint(0,T,(L,),device=dev,dtype=torch.int32)
Inter=torch.zeros(INTER,L,device=dev,dtype=torch.float32)
mW=from_dlpack(Wu); mX=from_dlpack(Xu); mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn))
meidx=from_dlpack(eidx); mtidx=from_dlpack(tidx); mInter=from_dlpack(Inter)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW,mX,mSFW,mSFX,meidx,mtidx,mInter,E,M2,HIDDEN,T,L,INTER,cur)
comp(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,cur); torch.cuda.synchronize()
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq32(p,s,KK):
    R=p.shape[0]; pu=p.view(torch.uint8).view(R,KK//8,4); v=torch.empty(R,KK,device=dev)
    for b in range(4):
        lo=(pu[:,:,b]&0xF).long(); hi=((pu[:,:,b]>>4)&0xF).long()
        v[:,(b*2)::8]=lut[lo]; v[:,(b*2+1)::8]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
Wf=deq32(Wu.reshape(E*M2,nw),SFW.reshape(E*M2,sfk),HIDDEN).reshape(E,M2,HIDDEN); Xf=deq32(Xu,SFX,HIDDEN)
ref=torch.empty(INTER,L,device=dev)
for l in range(L):
    g=Wf[eidx[l],:INTER]@Xf[tidx[l]]; u=Wf[eidx[l],INTER:]@Xf[tidx[l]]; ref[:,l]=(g*torch.sigmoid(g))*u
cos=torch.nn.functional.cosine_similarity(Inter.flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
print(f"GATEUP-WIN cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,cur)); wb=L*(M2*HIDDEN//2)
print(f"GATEUP-WIN E={E} L={L}: {t:.2f}us  BW(per-pair)={wb/(t*1e-6)/1e12:.2f}TB/s  (was 2131us/0.22)")

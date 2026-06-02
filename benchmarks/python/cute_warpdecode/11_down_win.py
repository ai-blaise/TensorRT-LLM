import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
TM=8; TK=32

@cute.kernel
def down_k(mW2, mI, mSFW2, mSFI, meidx, mtidx, mRW, mOut, HIDDEN: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tk,tm,_=cute.arch.thread_idx()
    e=meidx[bidz]; t=mtidx[bidz]; rw=mRW[bidz]
    W2e=mW2[e,None,None]; SFW2e=mSFW2[e,None,None]   # (HIDDEN, INTER//8 words) ; (HIDDEN, INTER//16 sf)
    Il=mI[bidz,None]; SFIl=mSFI[bidz,None]           # (INTER//8,) ; (INTER//16,)
    gW=cute.local_tile(W2e,cute.slice_(MMA,(None,0,None)),(None,None))   # (TM,1,H/TM,nw)
    gSFW=cute.local_tile(SFW2e,cute.slice_(MMA,(None,0,None)),(None,None))
    alloc=cutlass.utils.SmemAllocator()
    sh=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    acc=cutlass.Float32(0.0)
    nw=cute.size(gW.layout[3].shape)
    for kt in cutlass.range(tk, nw, TK):
        aw=gW[tm,None,bidx,kt].load()
        af=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(aw[0].ir_value()), (8,), cutlass.Float16)
        bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(Il[kt].ir_value()), (8,), cutlass.Float16)
        sfa=cutlass.Float32(gSFW[tm,0,bidx,kt//2]); sfb=cutlass.Float32(SFIl[kt//2])
        prod=(af*bf).to(cutlass.Float32)
        s=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): s=s+prod[i]
        acc=acc+s*sfa*sfb
    sh[tm,tk]=acc
    cute.arch.sync_threads()
    if tk==0:
        tot=cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(TK): tot=tot+sh[tm,r]
        h=bidx*TM+tm
        cute.arch.atomic_add(mOut.iterator+(t*HIDDEN+h), tot*rw)

@cute.jit
def launch(mW2,mI,mSFW2_raw,mSFI_raw,meidx,mtidx,mRW,mOut, E:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(TM,1,1); sfk=INTER//16
    mSFW2=cute.make_tensor(mSFW2_raw.iterator, cute.make_layout((E,HIDDEN,sfk),stride=(HIDDEN*sfk,sfk,1)))
    mSFI =cute.make_tensor(mSFI_raw.iterator, cute.make_layout((L,sfk),stride=(sfk,1)))
    down_k(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,HIDDEN,MMA).launch(grid=(cute.ceil_div(HIDDEN,TM),1,L),block=(TK,TM,1),stream=stream)

dev="cuda"; torch.manual_seed(1)
E,HIDDEN,INTER,T,L = 16,7168,2048,32,32; nw=INTER//8; sfk=INTER//16
W2u=torch.randint(0,2**31,(E,HIDDEN,nw),device=dev,dtype=torch.int32)
Iu=torch.randint(0,2**31,(L,nw),device=dev,dtype=torch.int32)
SFW2=torch.randint(1,15,(E,HIDDEN,sfk),device=dev,dtype=torch.uint8).contiguous()
SFI=torch.randint(1,15,(L,sfk),device=dev,dtype=torch.uint8).contiguous()
eidx=torch.randint(0,E,(L,),device=dev,dtype=torch.int32); tidx=torch.randint(0,T,(L,),device=dev,dtype=torch.int32); rw=torch.rand(L,device=dev,dtype=torch.float32)
Out=torch.zeros(T,HIDDEN,device=dev,dtype=torch.float32)
mW2=from_dlpack(W2u); mI=from_dlpack(Iu); mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mSFI=from_dlpack(SFI.view(torch.float8_e4m3fn))
meidx=from_dlpack(eidx); mtidx=from_dlpack(tidx); mRW=from_dlpack(rw); mOut=from_dlpack(Out)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,E,HIDDEN,INTER,T,L,cur)
comp(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,cur); torch.cuda.synchronize()
# reference
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq32(p,s,KK):  # p:(R,nw) int32, s:(R,sfk) -> (R,KK)
    R=p.shape[0]; pu=p.view(torch.uint8).view(R,nw,4); v=torch.empty(R,KK,device=dev)
    for b in range(4):
        lo=(pu[:,:,b]&0xF).long(); hi=((pu[:,:,b]>>4)&0xF).long()
        v[:,(b*2)::8]=lut[lo]; v[:,(b*2+1)::8]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
W2f=deq32(W2u.reshape(E*HIDDEN,nw),SFW2.reshape(E*HIDDEN,sfk),INTER).reshape(E,HIDDEN,INTER); If=deq32(Iu,SFI,INTER)
ref=torch.zeros(T,HIDDEN,device=dev)
for l in range(L): ref[tidx[l]] += (W2f[eidx[l]]@If[l])*rw[l].item()
cos=torch.nn.functional.cosine_similarity(Out.flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
print(f"DOWN-WIN cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,cur))
wb=L*(HIDDEN*INTER//2); eb=E*(HIDDEN*INTER//2)
print(f"DOWN-WIN E={E} L={L}: {t:.2f}us  BW(per-pair)={wb/(t*1e-6)/1e12:.2f}TB/s  (was 974us/0.24; expert-once-floor={eb/6.8e12*1e6:.1f}us)")

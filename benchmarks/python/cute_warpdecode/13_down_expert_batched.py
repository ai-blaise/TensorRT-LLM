import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
TM=8; TK=32

@cute.kernel
def down_eb(mW2, mI, mSFW2, mSFI, mPtr, mPairL, mTok, mRW, mOut, HIDDEN: cutlass.Constexpr, NWPT: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,e=cute.arch.block_idx(); tk,tm,_=cute.arch.thread_idx()
    W2e=mW2[e,None,None]; SFW2e=mSFW2[e,None,None]
    gW=cute.local_tile(W2e,cute.slice_(MMA,(None,0,None)),(None,None))     # (TM,1,H/TM,nw)
    gSFW=cute.local_tile(SFW2e,cute.slice_(MMA,(None,0,None)),(None,None))
    alloc=cutlass.utils.SmemAllocator()
    sh=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    # --- cache this thread's weight slice ONCE (decoded f16 + scales) ---
    cw=cute.make_fragment(NWPT*8, cutlass.Float16)
    cs=cute.make_fragment(NWPT, cutlass.Float32)
    for w in cutlass.range_constexpr(NWPT):
        kt=tk+w*TK
        aw=gW[tm,None,bidx,kt].load()
        af=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(aw[0].ir_value()), (8,), cutlass.Float16)
        for i in cutlass.range_constexpr(8): cw[w*8+i]=af[i]
        cs[w]=cutlass.Float32(gSFW[tm,0,bidx,kt//2])
    # --- loop this expert's routed pairs, reusing cached weight ---
    lo=mPtr[e]; hi=mPtr[e+1]
    for idx in cutlass.range(lo, hi, 1):
        l=mPairL[idx]; tok=mTok[idx]; rw=mRW[idx]
        Il=mI[l,None]; SFIl=mSFI[l,None]
        acc=cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(NWPT):
            kt=tk+w*TK
            bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(Il[kt].ir_value()), (8,), cutlass.Float16)
            sfb=cutlass.Float32(SFIl[kt//2])
            s=cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(8): s=s+cutlass.Float32(cw[w*8+i])*cutlass.Float32(bf[i])
            acc=acc+s*cs[w]*sfb
        sh[tm,tk]=acc
        cute.arch.sync_threads()
        if tk==0:
            tot=cutlass.Float32(0.0)
            for r in cutlass.range_constexpr(TK): tot=tot+sh[tm,r]
            h=bidx*TM+tm
            cute.arch.atomic_add(mOut.iterator+(tok*HIDDEN+h), tot*rw)
        cute.arch.sync_threads()

@cute.jit
def launch(mW2,mI,mSFW2_raw,mSFI_raw,mPtr,mPairL,mTok,mRW,mOut, E:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,INTER:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    nw=INTER//8; MMA=(TM,1,1); sfk=INTER//16; NWPT=nw//TK
    mSFW2=cute.make_tensor(mSFW2_raw.iterator, cute.make_layout((E,HIDDEN,sfk),stride=(HIDDEN*sfk,sfk,1)))
    mSFI =cute.make_tensor(mSFI_raw.iterator, cute.make_layout((L,sfk),stride=(sfk,1)))
    down_eb(mW2,mI,mSFW2,mSFI,mPtr,mPairL,mTok,mRW,mOut,HIDDEN,NWPT,MMA).launch(grid=(cute.ceil_div(HIDDEN,TM),1,E),block=(TK,TM,1),stream=stream)

dev="cuda"; torch.manual_seed(1)
E,HIDDEN,INTER,T,L = 16,7168,2048,32,32; nw=INTER//8; sfk=INTER//16
W2u=torch.randint(0,2**31,(E,HIDDEN,nw),device=dev,dtype=torch.int32); SFW2=torch.randint(1,15,(E,HIDDEN,sfk),device=dev,dtype=torch.uint8).contiguous()
Iu=torch.randint(0,2**31,(L,nw),device=dev,dtype=torch.int32); SFI=torch.randint(1,15,(L,sfk),device=dev,dtype=torch.uint8).contiguous()
eidx=torch.randint(0,E,(L,),device=dev,dtype=torch.int64); tok=torch.randint(0,T,(L,),device=dev,dtype=torch.int32); rw=torch.rand(L,device=dev,dtype=torch.float32)
# CSR group pairs by expert
order=torch.argsort(eidx); ecount=torch.bincount(eidx,minlength=E); ptr=torch.zeros(E+1,dtype=torch.int32,device=dev); ptr[1:]=torch.cumsum(ecount,0)
pairL=order.to(torch.int32); sortTok=tok[order].contiguous(); sortRW=rw[order].contiguous()
Out=torch.zeros(T,HIDDEN,device=dev,dtype=torch.float32)
mW2=from_dlpack(W2u); mI=from_dlpack(Iu); mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mSFI=from_dlpack(SFI.view(torch.float8_e4m3fn))
mPtr=from_dlpack(ptr); mPairL=from_dlpack(pairL); mTok=from_dlpack(sortTok); mRW=from_dlpack(sortRW); mOut=from_dlpack(Out)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW2,mI,mSFW2,mSFI,mPtr,mPairL,mTok,mRW,mOut,E,HIDDEN,INTER,L,cur)
comp(mW2,mI,mSFW2,mSFI,mPtr,mPairL,mTok,mRW,mOut,cur); torch.cuda.synchronize()
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq32(p,s,KK):
    R=p.shape[0]; pu=p.view(torch.uint8).view(R,KK//8,4); v=torch.empty(R,KK,device=dev)
    for b in range(4):
        lo2=(pu[:,:,b]&0xF).long(); hi2=((pu[:,:,b]>>4)&0xF).long()
        v[:,(b*2)::8]=lut[lo2]; v[:,(b*2+1)::8]=lut[hi2]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
W2f=deq32(W2u.reshape(E*HIDDEN,nw),SFW2.reshape(E*HIDDEN,sfk),INTER).reshape(E,HIDDEN,INTER); If=deq32(Iu,SFI,INTER)
ref=torch.zeros(T,HIDDEN,device=dev)
for l in range(L): ref[tok[l]] += (W2f[eidx[l]]@If[l])*rw[l].item()
cos=torch.nn.functional.cosine_similarity(Out.flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
print(f"DOWN-EB cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e2=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e2.record(); torch.cuda.synchronize(); return s.elapsed_time(e2)/it*1000
t=bench(lambda: comp(mW2,mI,mSFW2,mSFI,mPtr,mPairL,mTok,mRW,mOut,cur))
eb=E*(HIDDEN*INTER//2)
print(f"DOWN-EB E={E} L={L}: {t:.2f}us  expert-once-BW={eb/(t*1e-6)/1e12:.2f}TB/s  (per-pair down-win was 136us/1.72; floor {eb/6.8e12*1e6:.1f}us)")

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
    gW=cute.local_tile(We,cute.slice_(MMA,(None,0,None)),(None,None)); gSFW=cute.local_tile(SFWe,cute.slice_(MMA,(None,0,None)),(None,None))
    alloc=cutlass.utils.SmemAllocator()
    shg=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    shu=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    gacc=cutlass.Float32(0.0); uacc=cutlass.Float32(0.0); nw=cute.size(gW.layout[3].shape)
    for kt in cutlass.range(tk, nw, TK):
        bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(Xt[kt].ir_value()), (8,), cutlass.Float16); sfx=cutlass.Float32(SFXt[kt//2])
        gw=gW[tm,None,bidx,kt].load(); gaf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(gw[0].ir_value()), (8,), cutlass.Float16)
        gp=(gaf*bf).to(cutlass.Float32); gs=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): gs=gs+gp[i]
        gacc=gacc+gs*cutlass.Float32(gSFW[tm,0,bidx,kt//2])*sfx
        uw=gW[tm,None,bidx+UPOFF,kt].load(); uaf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(uw[0].ir_value()), (8,), cutlass.Float16)
        up=(uaf*bf).to(cutlass.Float32); us=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): us=us+up[i]
        uacc=uacc+us*cutlass.Float32(gSFW[tm,0,bidx+UPOFF,kt//2])*sfx
    shg[tm,tk]=gacc; shu[tm,tk]=uacc; cute.arch.sync_threads()
    if tk==0:
        g=cutlass.Float32(0.0); u=cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(TK): g=g+shg[tm,r]; u=u+shu[tm,r]
        silu=g*(cutlass.Float32(1.0)/(cutlass.Float32(1.0)+cute.arch.exp(-g)))
        mInter[bidx*TM+tm, bidz]=silu*u

@cute.jit
def launch_gu(mW,mX,mSFW_raw,mSFX_raw,meidx,mtidx,mInter, E:cutlass.Constexpr,M2:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr,INTER:cutlass.Constexpr, stream):
    MMA=(TM,1,1); sfk=HIDDEN//16
    mSFW=cute.make_tensor(mSFW_raw.iterator, cute.make_layout((E,M2,sfk),stride=(M2*sfk,sfk,1)))
    mSFX=cute.make_tensor(mSFX_raw.iterator, cute.make_layout((T,sfk),stride=(sfk,1)))
    gu_k(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,INTER//TM,MMA).launch(grid=(cute.ceil_div(INTER,TM),1,L),block=(TK,TM,1),stream=stream)

@cute.kernel
def dn_k(mW2, mI, mSFW2, mSFI, meidx, mtidx, mRW, mOut, HIDDEN: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tk,tm,_=cute.arch.thread_idx()
    e=meidx[bidz]; t=mtidx[bidz]; rw=mRW[bidz]
    W2e=mW2[e,None,None]; SFW2e=mSFW2[e,None,None]; Il=mI[bidz,None]; SFIl=mSFI[bidz,None]
    gW=cute.local_tile(W2e,cute.slice_(MMA,(None,0,None)),(None,None)); gSFW=cute.local_tile(SFW2e,cute.slice_(MMA,(None,0,None)),(None,None))
    alloc=cutlass.utils.SmemAllocator(); sh=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    acc=cutlass.Float32(0.0); nw=cute.size(gW.layout[3].shape)
    for kt in cutlass.range(tk, nw, TK):
        aw=gW[tm,None,bidx,kt].load()
        af=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(aw[0].ir_value()), (8,), cutlass.Float16)
        bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(Il[kt].ir_value()), (8,), cutlass.Float16)
        sfa=cutlass.Float32(gSFW[tm,0,bidx,kt//2]); sfb=cutlass.Float32(SFIl[kt//2])
        prod=(af*bf).to(cutlass.Float32); s=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): s=s+prod[i]
        acc=acc+s*sfa*sfb
    sh[tm,tk]=acc; cute.arch.sync_threads()
    if tk==0:
        tot=cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(TK): tot=tot+sh[tm,r]
        h=bidx*TM+tm; cute.arch.atomic_add(mOut.iterator+(t*HIDDEN+h), tot*rw)

@cute.jit
def launch_dn(mW2,mI,mSFW2_raw,mSFI_raw,meidx,mtidx,mRW,mOut, E:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(TM,1,1); sfk=INTER//16
    mSFW2=cute.make_tensor(mSFW2_raw.iterator, cute.make_layout((E,HIDDEN,sfk),stride=(HIDDEN*sfk,sfk,1)))
    mSFI =cute.make_tensor(mSFI_raw.iterator, cute.make_layout((L,sfk),stride=(sfk,1)))
    dn_k(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,HIDDEN,MMA).launch(grid=(cute.ceil_div(HIDDEN,TM),1,L),block=(TK,TM,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
HIDDEN,INTER=7168,2048; M2=2*INTER

def setup(E,T,L):
    nwH=HIDDEN//8; sfkH=HIDDEN//16; nwI=INTER//8; sfkI=INTER//16
    d={}
    d['Wu']=torch.randint(0,2**31,(E,M2,nwH),device=dev,dtype=torch.int32); d['SFW']=torch.randint(1,15,(E,M2,sfkH),device=dev,dtype=torch.uint8).contiguous()
    d['Xu']=torch.randint(0,2**31,(T,nwH),device=dev,dtype=torch.int32); d['SFX']=torch.randint(1,15,(T,sfkH),device=dev,dtype=torch.uint8).contiguous()
    d['W2u']=torch.randint(0,2**31,(E,HIDDEN,nwI),device=dev,dtype=torch.int32); d['SFW2']=torch.randint(1,15,(E,HIDDEN,sfkI),device=dev,dtype=torch.uint8).contiguous()
    d['eidx']=torch.randint(0,E,(L,),device=dev,dtype=torch.int32); d['tidx']=torch.randint(0,T,(L,),device=dev,dtype=torch.int32); d['rw']=torch.rand(L,device=dev,dtype=torch.float32)
    d['Inter']=torch.zeros(INTER,L,device=dev,dtype=torch.float32); d['Iu']=torch.randint(0,2**31,(L,nwI),device=dev,dtype=torch.int32); d['SFI']=torch.randint(1,15,(L,sfkI),device=dev,dtype=torch.uint8).contiguous()
    d['Out']=torch.zeros(T,HIDDEN,device=dev,dtype=torch.float32)
    return d

cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def build(E,T,L,d):
    mW=from_dlpack(d['Wu']); mX=from_dlpack(d['Xu']); mSFW=from_dlpack(d['SFW'].view(torch.float8_e4m3fn)); mSFX=from_dlpack(d['SFX'].view(torch.float8_e4m3fn))
    meidx=from_dlpack(d['eidx']); mtidx=from_dlpack(d['tidx']); mInter=from_dlpack(d['Inter'])
    cgu=cute.compile(launch_gu,mW,mX,mSFW,mSFX,meidx,mtidx,mInter,E,M2,HIDDEN,T,L,INTER,cur)
    mW2=from_dlpack(d['W2u']); mI=from_dlpack(d['Iu']); mSFW2=from_dlpack(d['SFW2'].view(torch.float8_e4m3fn)); mSFI=from_dlpack(d['SFI'].view(torch.float8_e4m3fn))
    mRW=from_dlpack(d['rw']); mOut=from_dlpack(d['Out'])
    cdn=cute.compile(launch_dn,mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,E,HIDDEN,INTER,T,L,cur)
    gu_args=(mW,mX,mSFW,mSFX,meidx,mtidx,mInter); dn_args=(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut)
    return cgu,cdn,gu_args,dn_args

def graph_time(cgu,cdn,gu_args,dn_args,nrep=20,it=50):
    def step(st):
        cgu(*gu_args,st); cdn(*dn_args,st)
    for _ in range(10): step(cur)
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        for _ in range(nrep): step(cap)
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it/nrep*1000

print("=== Optimized WarpDecode MoE — graph-captured production latency (per-rank decode) ===")
for (G,conc,E,T,L) in [(8,16,16,16,16),(8,32,16,32,32),(4,32,32,32,64),(2,32,64,32,128)]:
    d=setup(E,T,L); cgu,cdn,gu,dn=build(E,T,L,d)
    t=graph_time(cgu,cdn,gu,dn)
    eb=E*((M2*HIDDEN//2)+(HIDDEN*INTER//2))
    print(f"G={G} conc={conc:2d} (E={E:2d},L={L:3d}): MoE(gate_up+down) graph={t:7.2f}us | expert-once-floor={eb/6.8e12*1e6:6.1f}us")

import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
TM=32; TK=32; BKB=32   # threads_per_m, threads_per_k, bytes per K-tile (=64 FP4)

@cute.jit
def decode(nib):
    e=(nib>>1)&3; m=nib&1; sg=nib>>3; egt=(e|(e>>1))&1
    mf=cutlass.Float32(m); ef=cutlass.Float32(egt)
    base=mf+ef*(cutlass.Float32(1.0)-cutlass.Float32(0.5)*mf)
    pw=cutlass.Float32(0.5)*cutlass.Float32(1<<e)
    return base*pw*(cutlass.Float32(1.0)-cutlass.Float32(2.0)*cutlass.Float32(sg))

@cute.kernel
def gemv_x(mAu, mBu, mC, NB: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx()
    tk,tm,_=cute.arch.thread_idx()   # tk = K-slice (FAST/warp dim -> coalesced); tm = output row
    gA=cute.local_tile(mAu,cute.slice_(MMA,(None,0,None)),(None,None,None))
    gB=cute.local_tile(mBu,cute.slice_(MMA,(0,None,None)),(None,None,None))
    gC=cute.local_tile(mC,cute.slice_(MMA,(None,None,0)),(None,None,None))
    alloc=cutlass.utils.SmemAllocator()
    sh=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    acc=cutlass.Float32(0.0)
    ntiles=cute.size(gA.layout[3].shape)
    for kt in cutlass.range(tk, ntiles, TK):   # warp (tk 0..31) reads consecutive K-tiles of row tm -> coalesced
        au=gA[tm,None,bidx,kt,bidz].load(); bu=gB[0,None,0,kt,bidz].load()
        ra=cute.make_rmem_tensor_like(au); rb=cute.make_rmem_tensor_like(bu)
        ra.store(au); rb.store(bu)
        for j in cutlass.range_constexpr(NB):
            ab=cutlass.Int32(ra[j]); bb=cutlass.Int32(rb[j])
            acc = acc + decode(ab&0xF)*decode(bb&0xF) + decode((ab>>4)&0xF)*decode((bb>>4)&0xF)
    sh[tm,tk]=acc
    cute.arch.sync_threads()
    if tk==0:
        tot=cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(TK):
            tot=tot+sh[tm,r]
        gC[tm,None,bidx,0,bidz][0]=tot

@cute.jit
def launch(mAu,mBu,mC, M:cutlass.Constexpr,KB:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(TM,1,BKB)
    gemv_x(mAu,mBu,mC,BKB,MMA).launch(grid=(cute.ceil_div(M,TM),1,L),block=(TK,TM,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
M,K,L=4096,7168,8; KB=K//2
Au=torch.randint(0,256,(M,L,KB),device=dev,dtype=torch.uint8); Bu=torch.randint(0,256,(1,L,KB),device=dev,dtype=torch.uint8)
C=torch.zeros(M,1,L,device=dev,dtype=torch.float32)
mAu=from_dlpack(Au.permute(0,2,1)); mBu=from_dlpack(Bu.permute(0,2,1)); mC=from_dlpack(C)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mAu,mBu,mC,M,KB,L,cur)
comp(mAu,mBu,mC,cur); torch.cuda.synchronize()
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq(p):
    lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,K,device=dev); v[:,0::2]=lut[lo[:,0,:]]; v[:,1::2]=lut[hi[:,0,:]]; return v
ref=(deq(Au)@deq(Bu).T)[:,0]; cu=C[:,0,0]
cos=torch.nn.functional.cosine_similarity(cu.unsqueeze(0),ref.unsqueeze(0)).item()
print(f"X-GEMV cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mAu,mBu,mC,cur)); by=M*KB*L
print(f"X-GEMV (32x32 threads, SMEM reduce) M={M} K={K} L={L}: {t:.2f}us  BW={by/(t*1e-6)/1e12:.2f}TB/s  (was 555us/0.21; Veitner ~119us/0.98)")

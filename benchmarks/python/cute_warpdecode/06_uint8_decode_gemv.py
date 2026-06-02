import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
SVS=16; BKB=32   # bytes per K-tile (=64 FP4)

@cute.jit
def decode(nib):   # nib: Int32 0..15 -> Float32 e2m1 value (no transcendental, no select)
    e=(nib>>1)&3; m=nib&1; sg=nib>>3
    egt=(e|(e>>1))&1
    mf=cutlass.Float32(m); ef=cutlass.Float32(egt)
    base=mf+ef*(cutlass.Float32(1.0)-cutlass.Float32(0.5)*mf)
    pw=cutlass.Float32(0.5)*cutlass.Float32(1<<e)
    return base*pw*(cutlass.Float32(1.0)-cutlass.Float32(2.0)*cutlass.Float32(sg))

@cute.kernel
def gemv_u8(mAu, mBu, mC, NB: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,bidy,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx()
    gA=cute.local_tile(mAu,cute.slice_(MMA,(None,0,None)),(None,None,None))
    gB=cute.local_tile(mBu,cute.slice_(MMA,(0,None,None)),(None,None,None))
    gC=cute.local_tile(mC,cute.slice_(MMA,(None,None,0)),(None,None,None))
    tCgC=gC[tidx,None,bidx,bidy,bidz]
    frag=cute.make_fragment(1,cutlass.Float32); frag[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gA.layout[3].shape)):
        au=gA[tidx,None,bidx,kt,bidz].load()       # BKB uint8 (vectorized load)
        bu=gB[0,None,bidy,kt,bidz].load()
        ra=cute.make_rmem_tensor_like(au); rb=cute.make_rmem_tensor_like(bu)
        ra.store(au); rb.store(bu)
        acc=frag[0]
        for j in cutlass.range_constexpr(NB):
            abyte=cutlass.Int32(ra[j]); bbyte=cutlass.Int32(rb[j])
            alo=decode(abyte&0xF); ahi=decode((abyte>>4)&0xF)
            blo=decode(bbyte&0xF); bhi=decode((bbyte>>4)&0xF)
            acc = acc + alo*blo + ahi*bhi
        frag[0]=acc
    cute.autovec_copy(frag, tCgC)

@cute.jit
def launch(mAu,mBu,mC, M:cutlass.Constexpr,KB:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(128,1,BKB)
    gemv_u8(mAu,mBu,mC,BKB,MMA).launch(grid=(cute.ceil_div(M,128),1,L),block=(128,1,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
M,K,L=4096,7168,8; KB=K//2
Au=torch.randint(0,256,(M,L,KB),device=dev,dtype=torch.uint8)
Bu=torch.randint(0,256,(1,L,KB),device=dev,dtype=torch.uint8)
C=torch.zeros(M,1,L,device=dev,dtype=torch.float32)
mAu=from_dlpack(Au.permute(0,2,1)); mBu=from_dlpack(Bu.permute(0,2,1))   # (M,KB,L) uint8
mC=from_dlpack(C)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mAu,mBu,mC,M,KB,L,cur)
comp(mAu,mBu,mC,cur); torch.cuda.synchronize()
# reference
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq(p):  # (R,L,KB)->(R,K) L=fixed slice
    lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,K,device=dev); v[:,0::2]=lut[lo[:,0,:]]; v[:,1::2]=lut[hi[:,0,:]]
    return v
ref=(deq(Au)@deq(Bu).T)[:,0]; cu=C[:,0,0]
cos=torch.nn.functional.cosine_similarity(cu.unsqueeze(0),ref.unsqueeze(0)).item()
print(f"U8-GEMV cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mAu,mBu,mC,cur)); by=M*KB*L
print(f"U8-GEMV M={M} K={K} L={L}: {t:.2f}us  BW={by/(t*1e-6)/1e12:.2f}TB/s  (Float4 .load was 582us/0.20TB/s; Veitner 119us/0.98)")

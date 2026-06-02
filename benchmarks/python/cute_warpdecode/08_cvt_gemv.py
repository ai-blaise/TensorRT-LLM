import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
NW=8   # uint32 words per K-tile = 8*8 = 64 FP4

@cute.kernel
def gemv_cvt(mA32, mB32, mC, MMA: cutlass.Constexpr):
    bidx,bidy,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx()
    gA=cute.local_tile(mA32,cute.slice_(MMA,(None,0,None)),(None,None,None))
    gB=cute.local_tile(mB32,cute.slice_(MMA,(0,None,None)),(None,None,None))
    gC=cute.local_tile(mC,cute.slice_(MMA,(None,None,0)),(None,None,None))
    tCgC=gC[tidx,None,bidx,bidy,bidz]
    frag=cute.make_fragment(1,cutlass.Float32); frag[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gA.layout[3].shape)):
        aw=gA[tidx,None,bidx,kt,bidz].load(); bw=gB[0,None,bidy,kt,bidz].load()
        ra=cute.make_rmem_tensor_like(aw); rb=cute.make_rmem_tensor_like(bw)
        ra.store(aw); rb.store(bw)
        acc=frag[0]
        for w in cutlass.range_constexpr(NW):
            af=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(ra[w].ir_value()), (8,), cutlass.Float16)
            bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(rb[w].ir_value()), (8,), cutlass.Float16)
            prod=(af*bf).to(cutlass.Float32)
            for i in cutlass.range_constexpr(8):
                acc=acc+prod[i]
        frag[0]=acc
    cute.autovec_copy(frag, tCgC)

@cute.jit
def launch(mA32,mB32,mC, M:cutlass.Constexpr,KW:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(128,1,NW)
    gemv_cvt(mA32,mB32,mC,MMA).launch(grid=(cute.ceil_div(M,128),1,L),block=(128,1,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
M,K,L=4096,7168,8; KW=K//8   # uint32 words along K
Au=torch.randint(0,2**31,(M,L,KW),device=dev,dtype=torch.int32)
Bu=torch.randint(0,2**31,(1,L,KW),device=dev,dtype=torch.int32)
C=torch.zeros(M,1,L,device=dev,dtype=torch.float32)
mA32=from_dlpack(Au.permute(0,2,1)); mB32=from_dlpack(Bu.permute(0,2,1)); mC=from_dlpack(C)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mA32,mB32,mC,M,KW,L,cur)
comp(mA32,mB32,mC,cur); torch.cuda.synchronize()
# reference: decode the int32 words to fp4 values (8 per word) and dot
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq32(p):   # p:(R,L,KW) int32 -> (R,K) for L-slice0; each word=4 bytes=8 nibbles, byte i -> lo,hi
    R=p.shape[0]; pu=p[:,0,:].view(torch.uint8).view(R,KW,4)  # (R,KW,4 bytes)
    v=torch.empty(R,K,device=dev)
    for b in range(4):
        lo=(pu[:,:,b]&0xF).long(); hi=((pu[:,:,b]>>4)&0xF).long()
        v[:, (b*2)::8]=lut[lo]; v[:, (b*2+1)::8]=lut[hi]
    return v
ref=(deq32(Au)@deq32(Bu).T)[:,0]; cu=C[:,0,0]
cos=torch.nn.functional.cosine_similarity(cu.unsqueeze(0),ref.unsqueeze(0)).item()
print(f"CVT-GEMV cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mA32,mB32,mC,cur)); by=M*(K//2)*L
print(f"CVT-GEMV M={M} K={K} L={L}: {t:.2f}us  BW={by/(t*1e-6)/1e12:.2f}TB/s  (manual-decode was 555us/0.21; Veitner 119us/0.98)")

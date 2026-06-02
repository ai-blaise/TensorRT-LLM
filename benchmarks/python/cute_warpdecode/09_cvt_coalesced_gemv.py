import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
TM=8; TK=32   # 8 rows x 32 K-slice threads = 256 threads/block

@cute.kernel
def gemv_cx(mA32, mB32, mC, MMA: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tk,tm,_=cute.arch.thread_idx()
    gA=cute.local_tile(mA32,cute.slice_(MMA,(None,0,None)),(None,None,None))
    gB=cute.local_tile(mB32,cute.slice_(MMA,(0,None,None)),(None,None,None))
    gC=cute.local_tile(mC,cute.slice_(MMA,(None,None,0)),(None,None,None))
    alloc=cutlass.utils.SmemAllocator()
    sh=alloc.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_layout((TM,TK)))
    acc=cutlass.Float32(0.0)
    nw=cute.size(gA.layout[3].shape)
    for kt in cutlass.range(tk, nw, TK):   # warp(tk) reads consecutive uint32 words -> coalesced
        aw=gA[tm,None,bidx,kt,bidz].load(); bw=gB[0,None,0,kt,bidz].load()
        af=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(aw[0].ir_value()), (8,), cutlass.Float16)
        bf=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(bw[0].ir_value()), (8,), cutlass.Float16)
        prod=(af*bf).to(cutlass.Float32)
        for i in cutlass.range_constexpr(8):
            acc=acc+prod[i]
    sh[tm,tk]=acc
    cute.arch.sync_threads()
    if tk==0:
        tot=cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(TK):
            tot=tot+sh[tm,r]
        gC[tm,None,bidx,0,bidz][0]=tot

@cute.jit
def launch(mA32,mB32,mC, M:cutlass.Constexpr,KW:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(TM,1,1)
    gemv_cx(mA32,mB32,mC,MMA).launch(grid=(cute.ceil_div(M,TM),1,L),block=(TK,TM,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
M,K,L=4096,7168,8; KW=K//8
Au=torch.randint(0,2**31,(M,L,KW),device=dev,dtype=torch.int32); Bu=torch.randint(0,2**31,(1,L,KW),device=dev,dtype=torch.int32)
C=torch.zeros(M,1,L,device=dev,dtype=torch.float32)
mA32=from_dlpack(Au.permute(0,2,1)); mB32=from_dlpack(Bu.permute(0,2,1)); mC=from_dlpack(C)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mA32,mB32,mC,M,KW,L,cur)
comp(mA32,mB32,mC,cur); torch.cuda.synchronize()
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq32(p):
    R=p.shape[0]; pu=p[:,0,:].view(torch.uint8).view(R,KW,4); v=torch.empty(R,K,device=dev)
    for b in range(4):
        lo=(pu[:,:,b]&0xF).long(); hi=((pu[:,:,b]>>4)&0xF).long()
        v[:,(b*2)::8]=lut[lo]; v[:,(b*2+1)::8]=lut[hi]
    return v
ref=(deq32(Au)@deq32(Bu).T)[:,0]; cu=C[:,0,0]
cos=torch.nn.functional.cosine_similarity(cu.unsqueeze(0),ref.unsqueeze(0)).item()
print(f"CVT-X-GEMV cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mA32,mB32,mC,cur)); by=M*(K//2)*L
print(f"CVT-X-GEMV (cvt+coalesced+SMEM) M={M} K={K} L={L}: {t:.2f}us  BW={by/(t*1e-6)/1e12:.2f}TB/s  (per-row cvt was 130us/0.91; Veitner improved ~2.1)")

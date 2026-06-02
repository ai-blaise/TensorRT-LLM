import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
SVS=16; BK=64

@cute.kernel
def gemv_k(mA, mB, mSFA, mSFB, mC, MMA: cutlass.Constexpr):
    bidx,bidy,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx()
    gA=cute.local_tile(mA,cute.slice_(MMA,(None,0,None)),(None,None,None))
    gSFA=cute.local_tile(mSFA,cute.slice_(MMA,(None,0,None)),(None,None,None))
    gB=cute.local_tile(mB,cute.slice_(MMA,(0,None,None)),(None,None,None))
    gSFB=cute.local_tile(mSFB,cute.slice_(MMA,(0,None,None)),(None,None,None))
    gC=cute.local_tile(mC,cute.slice_(MMA,(None,None,0)),(None,None,None))
    tCgC=gC[tidx,None,bidx,bidy,bidz]
    frag=cute.make_fragment(1, cutlass.Float32); frag[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gA.layout[3].shape)):
        a=gA[tidx,None,bidx,kt,bidz].load().to(cutlass.Float32)
        b=gB[0,None,bidy,kt,bidz].load().to(cutlass.Float32)
        sa=gSFA[tidx,None,bidx,kt,bidz].load().to(cutlass.Float32)
        sb=gSFB[0,None,bidy,kt,bidz].load().to(cutlass.Float32)
        ra=cute.make_rmem_tensor_like(a);rb=cute.make_rmem_tensor_like(b)
        rsa=cute.make_rmem_tensor_like(sa);rsb=cute.make_rmem_tensor_like(sb)
        ra.store(a);rb.store(b);rsa.store(sa);rsb.store(sb)
        acc=frag[0]
        for i in cutlass.range_constexpr(MMA[2]):
            acc = acc + ra[i]*rsa[i]*rb[i]*rsb[i]
        frag[0]=acc
    cute.autovec_copy(frag, tCgC)

@cute.jit
def launch(mA,mB,mSFA_raw,mSFB_raw,mC, M:cutlass.Constexpr,K:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(128,1,BK); ks=K//SVS
    # explicit K-broadcast layout: logical (M,(16,ks),L) -> physical (M,ks,L) contiguous
    la=cute.make_layout((M,(SVS,ks),L), stride=(ks*L,(0,L),1))
    lb=cute.make_layout((1,(SVS,ks),L), stride=(ks*L,(0,L),1))
    mSFA=cute.make_tensor(mSFA_raw.iterator, la)
    mSFB=cute.make_tensor(mSFB_raw.iterator, lb)
    gemv_k(mA,mB,mSFA,mSFB,mC,MMA).launch(grid=(cute.ceil_div(M,128),1,L),block=(128,1,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
M,K,L=256,128,1; ks=K//SVS
Au=torch.randint(0,256,(M,L,K//2),device=dev,dtype=torch.uint8)
Bu=torch.randint(0,256,(1,L,K//2),device=dev,dtype=torch.uint8)
SFA=torch.randint(1,15,(M,ks,L),device=dev,dtype=torch.uint8).contiguous()
SFB=torch.randint(1,15,(1,ks,L),device=dev,dtype=torch.uint8).contiguous()
C=torch.zeros(M,1,L,device=dev,dtype=torch.float32)
mA=from_dlpack(Au.view(torch.float4_e2m1fn_x2).permute(0,2,1))
mB=from_dlpack(Bu.view(torch.float4_e2m1fn_x2).permute(0,2,1))
mSFA=from_dlpack(SFA.view(torch.float8_e4m3fn)); mSFB=from_dlpack(SFB.view(torch.float8_e4m3fn))
mC=from_dlpack(C)
print("mA",mA.element_type,tuple(mA.shape),"mSFA",mSFA.element_type,tuple(mSFA.shape))
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mA,mB,mSFA,mSFB,mC,M,K,L,cur)
comp(mA,mB,mSFA,mSFB,mC,cur); torch.cuda.synchronize()

# reference
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq(p,s):
    lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,K,device=dev); v[:,0::2]=lut[lo[:,0,:]]; v[:,1::2]=lut[hi[:,0,:]]
    sf=s.view(torch.float8_e4m3fn)[:,:,0].float().repeat_interleave(SVS,dim=1)
    return v*sf
ref=(deq(Au,SFA)@deq(Bu,SFB).T)[:,0]; cu=C[:,0,0]
cos=torch.nn.functional.cosine_similarity(cu.unsqueeze(0),ref.unsqueeze(0)).item()
rel=(cu-ref).norm().item()/(ref.norm().item()+1e-9)
print(f"GEMV cosine={cos:.6f} relerr={rel:.6f} -> {'PASS' if cos>0.999 and rel<0.02 else 'FAIL'}")
print("cu[:4]",[round(x,4) for x in cu[:4].tolist()]," ref[:4]",[round(x,4) for x in ref[:4].tolist()])

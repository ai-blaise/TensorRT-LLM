import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
SVS=16; BK=64

@cute.kernel
def down_k(mW2, mI, mSFW2, mSFI, meidx, mtidx, mRW, mOut, HIDDEN: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz = cute.arch.block_idx(); tidx,_,_ = cute.arch.thread_idx()
    e = meidx[bidz]; t = mtidx[bidz]; rw = mRW[bidz]
    W2e = mW2[e,None,None]; SFW2e = mSFW2[e,None,None]   # (HIDDEN, INTER)
    Il  = mI[bidz,None];    SFIl  = mSFI[bidz,None]       # (INTER,) for pair l=bidz
    gW   = cute.local_tile(W2e,  cute.slice_(MMA,(None,0,None)), (None,None))  # (128,BK,HIDDEN/128,INTER/BK)
    gSFW = cute.local_tile(SFW2e,cute.slice_(MMA,(None,0,None)), (None,None))
    gI   = cute.local_tile(Il,   (BK,), (None,))
    gSFI = cute.local_tile(SFIl, (BK,), (None,))
    frag = cute.make_fragment(1, cutlass.Float32); frag[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gW.layout[3].shape)):
        a  = gW[tidx,None,bidx,kt].load().to(cutlass.Float32)
        b  = gI[None,kt].load().to(cutlass.Float32)
        sa = gSFW[tidx,None,bidx,kt].load().to(cutlass.Float32)
        sb = gSFI[None,kt].load().to(cutlass.Float32)
        ra=cute.make_rmem_tensor_like(a);rb=cute.make_rmem_tensor_like(b)
        rsa=cute.make_rmem_tensor_like(sa);rsb=cute.make_rmem_tensor_like(sb)
        ra.store(a);rb.store(b);rsa.store(sa);rsb.store(sb)
        acc=frag[0]
        for i in cutlass.range_constexpr(MMA[2]):
            acc = acc + ra[i]*rsa[i]*rb[i]*rsb[i]
        frag[0]=acc
    h = bidx*128 + tidx
    contrib = frag[0] * rw
    ptr = mOut.iterator + (t*HIDDEN + h)
    cute.arch.atomic_add(ptr, contrib)

@cute.jit
def launch(mW2,mI,mSFW2_raw,mSFI_raw,meidx,mtidx,mRW,mOut, E:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(128,1,BK); ks=INTER//SVS
    lw=cute.make_layout((E,HIDDEN,(SVS,ks)), stride=(HIDDEN*ks,ks,(0,1)))
    li=cute.make_layout((L,(SVS,ks)),        stride=(ks,(0,1)))
    mSFW2=cute.make_tensor(mSFW2_raw.iterator, lw)
    mSFI =cute.make_tensor(mSFI_raw.iterator, li)
    down_k(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,HIDDEN,MMA).launch(grid=(cute.ceil_div(HIDDEN,128),1,L),block=(128,1,1),stream=stream)

dev="cuda"; torch.manual_seed(1)
E,HIDDEN,INTER,T,L = 8,7168,2048,8,64; ks=INTER//SVS
W2u = torch.randint(0,256,(E,HIDDEN,INTER//2),device=dev,dtype=torch.uint8)
Iu  = torch.randint(0,256,(L,INTER//2),device=dev,dtype=torch.uint8)
SFW2= torch.randint(1,15,(E,HIDDEN,ks),device=dev,dtype=torch.uint8).contiguous()
SFI = torch.randint(1,15,(L,ks),device=dev,dtype=torch.uint8).contiguous()
eidx= torch.randint(0,E,(L,),device=dev,dtype=torch.int32); tidx=torch.randint(0,T,(L,),device=dev,dtype=torch.int32)
rw  = torch.rand(L,device=dev,dtype=torch.float32)
Out = torch.zeros(T,HIDDEN,device=dev,dtype=torch.float32)
mW2=from_dlpack(W2u.view(torch.float4_e2m1fn_x2)); mI=from_dlpack(Iu.view(torch.float4_e2m1fn_x2))
mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mSFI=from_dlpack(SFI.view(torch.float8_e4m3fn))
meidx=from_dlpack(eidx); mtidx=from_dlpack(tidx); mRW=from_dlpack(rw); mOut=from_dlpack(Out)
print("mW2",mW2.element_type,tuple(mW2.shape))
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,E,HIDDEN,INTER,T,L,cur)
comp(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,cur); torch.cuda.synchronize()

E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq2(p,s,kk):
    lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,kk,device=dev); v[:,0::2]=lut[lo]; v[:,1::2]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(SVS,dim=1)
W2f=deq2(W2u.reshape(E*HIDDEN,INTER//2),SFW2.reshape(E*HIDDEN,ks),INTER).reshape(E,HIDDEN,INTER)
If =deq2(Iu,SFI,INTER)
ref=torch.zeros(T,HIDDEN,device=dev)
for l in range(L):
    ref[tidx[l]] += (W2f[eidx[l]]@If[l]) * rw[l].item()
cos=torch.nn.functional.cosine_similarity(Out.flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
reln=(Out-ref).norm().item()/(ref.norm().item()+1e-9)
print(f"DOWN+scatter cosine={cos:.6f} relerr={reln:.6f} -> {'PASS' if cos>0.999 and reln<0.02 else 'FAIL'}")
print("Out[0,:3]",[round(x,3) for x in Out[0,:3].tolist()]," ref",[round(x,3) for x in ref[0,:3].tolist()])

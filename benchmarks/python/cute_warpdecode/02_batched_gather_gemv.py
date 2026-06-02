import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
SVS=16; BK=64

@cute.kernel
def bgemv_k(mW, mX, mSFW, mSFX, meidx, mtidx, mC, MMA: cutlass.Constexpr):
    bidx,_,bidz = cute.arch.block_idx(); tidx,_,_ = cute.arch.thread_idx()
    e = meidx[bidz]; t = mtidx[bidz]
    # slice expert weight (E,M,K)->(M,K), token act (T,K)->(K,), and their SFs by runtime index
    We = mW[e,None,None]; SFWe = mSFW[e,None,None]   # (M,K) logical (None keeps mode)
    Xt = mX[t,None];      SFXt = mSFX[t,None]        # (K,) logical
    gW   = cute.local_tile(We,   cute.slice_(MMA,(None,0,None)), (None,None))  # (128,BK,M/128,K/BK)
    gSFW = cute.local_tile(SFWe, cute.slice_(MMA,(None,0,None)), (None,None))
    gX   = cute.local_tile(Xt,   (BK,), (None,))   # (BK, K/BK)
    gSFX = cute.local_tile(SFXt, (BK,), (None,))
    gC   = cute.local_tile(mC, (128,1), (None,None))  # (128,1,M/128,L)
    tCgC = gC[tidx,None,bidx,bidz]
    frag = cute.make_fragment(1, cutlass.Float32); frag[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gW.layout[3].shape)):  # mode 3 = K/BK tile count
        a  = gW[tidx,None,bidx,kt].load().to(cutlass.Float32)
        b  = gX[None,kt].load().to(cutlass.Float32)
        sa = gSFW[tidx,None,bidx,kt].load().to(cutlass.Float32)
        sb = gSFX[None,kt].load().to(cutlass.Float32)
        # a,b,sa,sb each length BK
        ra=cute.make_rmem_tensor_like(a);rb=cute.make_rmem_tensor_like(b)
        rsa=cute.make_rmem_tensor_like(sa);rsb=cute.make_rmem_tensor_like(sb)
        ra.store(a);rb.store(b);rsa.store(sa);rsb.store(sb)
        acc=frag[0]
        for i in cutlass.range_constexpr(MMA[2]):
            acc = acc + ra[i]*rsa[i]*rb[i]*rsb[i]
        frag[0]=acc
    cute.autovec_copy(frag, tCgC)

@cute.jit
def launch(mW,mX,mSFW_raw,mSFX_raw,meidx,mtidx,mC, E:cutlass.Constexpr,M:cutlass.Constexpr,K:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(128,1,BK); ks=K//SVS
    lw=cute.make_layout((E,M,(SVS,ks)), stride=(M*ks,ks,(0,1)))
    lx=cute.make_layout((T,(SVS,ks)),   stride=(ks,(0,1)))
    mSFW=cute.make_tensor(mSFW_raw.iterator, lw)
    mSFX=cute.make_tensor(mSFX_raw.iterator, lx)
    bgemv_k(mW,mX,mSFW,mSFX,meidx,mtidx,mC,MMA).launch(grid=(cute.ceil_div(M,128),1,L),block=(128,1,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
E,M,K,T,L = 4,128,128,4,8; ks=K//SVS
Wu  = torch.randint(0,256,(E,M,K//2),device=dev,dtype=torch.uint8)
Xu  = torch.randint(0,256,(T,K//2),device=dev,dtype=torch.uint8)
SFW = torch.randint(1,15,(E,M,ks),device=dev,dtype=torch.uint8).contiguous()
SFX = torch.randint(1,15,(T,ks),device=dev,dtype=torch.uint8).contiguous()
eidx= torch.randint(0,E,(L,),device=dev,dtype=torch.int32)
tidx= torch.randint(0,T,(L,),device=dev,dtype=torch.int32)
C   = torch.zeros(M,L,device=dev,dtype=torch.float32)
mW = from_dlpack(Wu.view(torch.float4_e2m1fn_x2))   # (E,M,K)
mX = from_dlpack(Xu.view(torch.float4_e2m1fn_x2))   # (T,K)
mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn))
mC = from_dlpack(C)
meidx=from_dlpack(eidx); mtidx=from_dlpack(tidx)
print("mW",mW.element_type,tuple(mW.shape),"mX",tuple(mX.shape))
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW,mX,mSFW,mSFX,meidx,mtidx,mC,E,M,K,T,L,cur)
comp(mW,mX,mSFW,mSFX,meidx,mtidx,mC,cur); torch.cuda.synchronize()

# reference
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq2(p,s):  # p:(R,K/2) s:(R,ks) -> (R,K)
    lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,K,device=dev); v[:,0::2]=lut[lo]; v[:,1::2]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(SVS,dim=1)
Wf=deq2(Wu.reshape(E*M,K//2),SFW.reshape(E*M,ks)).reshape(E,M,K)
Xf=deq2(Xu,SFX)
ref=torch.empty(M,L,device=dev)
for l in range(L):
    ref[:,l]=Wf[eidx[l]]@Xf[tidx[l]]
cos=torch.nn.functional.cosine_similarity(C.flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
rel=(C-ref).norm().item()/(ref.norm().item()+1e-9)
print(f"BATCHED-GEMV cosine={cos:.6f} relerr={rel:.6f} -> {'PASS' if cos>0.999 and rel<0.02 else 'FAIL'}")
print("C[:3,0]",[round(x,4) for x in C[:3,0].tolist()]," ref[:3,0]",[round(x,4) for x in ref[:3,0].tolist()])

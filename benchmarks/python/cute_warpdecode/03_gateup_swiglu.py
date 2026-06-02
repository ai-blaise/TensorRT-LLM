import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
SVS=16; BK=64

@cute.kernel
def gateup_k(mW, mX, mSFW, mSFX, meidx, mtidx, mInter, INTER128: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz = cute.arch.block_idx(); tidx,_,_ = cute.arch.thread_idx()
    e = meidx[bidz]; t = mtidx[bidz]
    We = mW[e,None,None]; SFWe = mSFW[e,None,None]   # (2*INTER, HIDDEN)
    Xt = mX[t,None];      SFXt = mSFX[t,None]         # (HIDDEN,)
    gW   = cute.local_tile(We,   cute.slice_(MMA,(None,0,None)), (None,None))  # (128,BK,2*INTER/128,K/BK)
    gSFW = cute.local_tile(SFWe, cute.slice_(MMA,(None,0,None)), (None,None))
    gX   = cute.local_tile(Xt,   (BK,), (None,))
    gSFX = cute.local_tile(SFXt, (BK,), (None,))
    gI   = cute.local_tile(mInter, (128,1), (None,None))  # (128,1,INTER/128,L)
    tIgI = gI[tidx,None,bidx,bidz]
    fg = cute.make_fragment(1, cutlass.Float32); fg[0]=cutlass.Float32(0.0)
    fu = cute.make_fragment(1, cutlass.Float32); fu[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gW.layout[3].shape)):
        ag  = gW[tidx,None,bidx,kt].load().to(cutlass.Float32)              # gate row j
        au  = gW[tidx,None,bidx+INTER128,kt].load().to(cutlass.Float32)     # up row j+INTER
        b   = gX[None,kt].load().to(cutlass.Float32)
        sag = gSFW[tidx,None,bidx,kt].load().to(cutlass.Float32)
        sau = gSFW[tidx,None,bidx+INTER128,kt].load().to(cutlass.Float32)
        sb  = gSFX[None,kt].load().to(cutlass.Float32)
        rag=cute.make_rmem_tensor_like(ag);rau=cute.make_rmem_tensor_like(au);rb=cute.make_rmem_tensor_like(b)
        rsag=cute.make_rmem_tensor_like(sag);rsau=cute.make_rmem_tensor_like(sau);rsb=cute.make_rmem_tensor_like(sb)
        rag.store(ag);rau.store(au);rb.store(b);rsag.store(sag);rsau.store(sau);rsb.store(sb)
        accg=fg[0]; accu=fu[0]
        for i in cutlass.range_constexpr(MMA[2]):
            bv = rb[i]*rsb[i]
            accg = accg + rag[i]*rsag[i]*bv
            accu = accu + rau[i]*rsau[i]*bv
        fg[0]=accg; fu[0]=accu
    g = fg[0]; u = fu[0]
    silu = g * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.arch.exp(-g)))
    out = cute.make_fragment(1, cutlass.Float32); out[0] = silu * u
    cute.autovec_copy(out, tIgI)

@cute.jit
def launch(mW,mX,mSFW_raw,mSFX_raw,meidx,mtidx,mInter, E:cutlass.Constexpr,M2:cutlass.Constexpr,K:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr,INTER:cutlass.Constexpr, stream):
    MMA=(128,1,BK); ks=K//SVS
    lw=cute.make_layout((E,M2,(SVS,ks)), stride=(M2*ks,ks,(0,1)))
    lx=cute.make_layout((T,(SVS,ks)),   stride=(ks,(0,1)))
    mSFW=cute.make_tensor(mSFW_raw.iterator, lw)
    mSFX=cute.make_tensor(mSFX_raw.iterator, lx)
    gateup_k(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,INTER//128,MMA).launch(grid=(cute.ceil_div(INTER,128),1,L),block=(128,1,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
E,INTER,K,T,L = 8,2048,7168,8,64; M2=2*INTER; ks=K//SVS
Wu = torch.randint(0,256,(E,M2,K//2),device=dev,dtype=torch.uint8)
Xu = torch.randint(0,256,(T,K//2),device=dev,dtype=torch.uint8)
SFW= torch.randint(1,15,(E,M2,ks),device=dev,dtype=torch.uint8).contiguous()
SFX= torch.randint(1,15,(T,ks),device=dev,dtype=torch.uint8).contiguous()
eidx=torch.randint(0,E,(L,),device=dev,dtype=torch.int32); tidx=torch.randint(0,T,(L,),device=dev,dtype=torch.int32)
Inter=torch.zeros(INTER,L,device=dev,dtype=torch.float32)
mW=from_dlpack(Wu.view(torch.float4_e2m1fn_x2)); mX=from_dlpack(Xu.view(torch.float4_e2m1fn_x2))
mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn))
meidx=from_dlpack(eidx); mtidx=from_dlpack(tidx); mInter=from_dlpack(Inter)
print("mW",mW.element_type,tuple(mW.shape))
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW,mX,mSFW,mSFX,meidx,mtidx,mInter,E,M2,K,T,L,INTER,cur)
comp(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,cur); torch.cuda.synchronize()

E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq2(p,s,kk):
    lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,kk,device=dev); v[:,0::2]=lut[lo]; v[:,1::2]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(SVS,dim=1)
Wf=deq2(Wu.reshape(E*M2,K//2),SFW.reshape(E*M2,ks),K).reshape(E,M2,K)
Xf=deq2(Xu,SFX,K)
ref=torch.empty(INTER,L,device=dev)
for l in range(L):
    g=Wf[eidx[l],:INTER]@Xf[tidx[l]]; u=Wf[eidx[l],INTER:]@Xf[tidx[l]]
    ref[:,l]=(g*torch.sigmoid(g))*u
cos=torch.nn.functional.cosine_similarity(Inter.flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
rel=(Inter-ref).norm().item()/(ref.norm().item()+1e-9)
print(f"GATE_UP+SwiGLU cosine={cos:.6f} relerr={rel:.6f} -> {'PASS' if cos>0.999 and rel<0.02 else 'FAIL'}")
print("I[:3,0]",[round(x,4) for x in Inter[:3,0].tolist()]," ref",[round(x,4) for x in ref[:3,0].tolist()])

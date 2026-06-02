import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
SVS=16; BK=64
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]

# ---------- gate_up + SwiGLU kernel (outputs Inter (INTER, L) fp32) ----------
@cute.kernel
def gateup_k(mW, mX, mSFW, mSFX, meidx, mtidx, mInter, INTER128: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz = cute.arch.block_idx(); tidx,_,_ = cute.arch.thread_idx()
    e = meidx[bidz]; t = mtidx[bidz]
    We = mW[e,None,None]; SFWe = mSFW[e,None,None]; Xt = mX[t,None]; SFXt = mSFX[t,None]
    gW   = cute.local_tile(We,   cute.slice_(MMA,(None,0,None)), (None,None))
    gSFW = cute.local_tile(SFWe, cute.slice_(MMA,(None,0,None)), (None,None))
    gX   = cute.local_tile(Xt,   (BK,), (None,)); gSFX = cute.local_tile(SFXt, (BK,), (None,))
    gI   = cute.local_tile(mInter, (128,1), (None,None)); tIgI = gI[tidx,None,bidx,bidz]
    fg = cute.make_fragment(1, cutlass.Float32); fg[0]=cutlass.Float32(0.0)
    fu = cute.make_fragment(1, cutlass.Float32); fu[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gW.layout[3].shape)):
        ag=gW[tidx,None,bidx,kt].load().to(cutlass.Float32); au=gW[tidx,None,bidx+INTER128,kt].load().to(cutlass.Float32)
        b=gX[None,kt].load().to(cutlass.Float32)
        sag=gSFW[tidx,None,bidx,kt].load().to(cutlass.Float32); sau=gSFW[tidx,None,bidx+INTER128,kt].load().to(cutlass.Float32)
        sb=gSFX[None,kt].load().to(cutlass.Float32)
        rag=cute.make_rmem_tensor_like(ag);rau=cute.make_rmem_tensor_like(au);rb=cute.make_rmem_tensor_like(b)
        rsag=cute.make_rmem_tensor_like(sag);rsau=cute.make_rmem_tensor_like(sau);rsb=cute.make_rmem_tensor_like(sb)
        rag.store(ag);rau.store(au);rb.store(b);rsag.store(sag);rsau.store(sau);rsb.store(sb)
        accg=fg[0]; accu=fu[0]
        for i in cutlass.range_constexpr(MMA[2]):
            bv=rb[i]*rsb[i]; accg=accg+rag[i]*rsag[i]*bv; accu=accu+rau[i]*rsau[i]*bv
        fg[0]=accg; fu[0]=accu
    g=fg[0]; u=fu[0]
    silu = g*(cutlass.Float32(1.0)/(cutlass.Float32(1.0)+cute.arch.exp(-g)))
    out=cute.make_fragment(1,cutlass.Float32); out[0]=silu*u; cute.autovec_copy(out, tIgI)

@cute.jit
def launch_gu(mW,mX,mSFW_raw,mSFX_raw,meidx,mtidx,mInter, E:cutlass.Constexpr,M2:cutlass.Constexpr,K:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr,INTER:cutlass.Constexpr, stream):
    MMA=(128,1,BK); ks=K//SVS
    mSFW=cute.make_tensor(mSFW_raw.iterator, cute.make_layout((E,M2,(SVS,ks)),stride=(M2*ks,ks,(0,1))))
    mSFX=cute.make_tensor(mSFX_raw.iterator, cute.make_layout((T,(SVS,ks)),stride=(ks,(0,1))))
    gateup_k(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,INTER//128,MMA).launch(grid=(cute.ceil_div(INTER,128),1,L),block=(128,1,1),stream=stream)

# ---------- down + scatter kernel ----------
@cute.kernel
def down_k(mW2, mI, mSFW2, mSFI, meidx, mtidx, mRW, mOut, HIDDEN: cutlass.Constexpr, MMA: cutlass.Constexpr):
    bidx,_,bidz = cute.arch.block_idx(); tidx,_,_ = cute.arch.thread_idx()
    e=meidx[bidz]; t=mtidx[bidz]; rw=mRW[bidz]
    W2e=mW2[e,None,None]; SFW2e=mSFW2[e,None,None]; Il=mI[bidz,None]; SFIl=mSFI[bidz,None]
    gW=cute.local_tile(W2e,cute.slice_(MMA,(None,0,None)),(None,None)); gSFW=cute.local_tile(SFW2e,cute.slice_(MMA,(None,0,None)),(None,None))
    gI=cute.local_tile(Il,(BK,),(None,)); gSFI=cute.local_tile(SFIl,(BK,),(None,))
    frag=cute.make_fragment(1,cutlass.Float32); frag[0]=cutlass.Float32(0.0)
    for kt in cutlass.range(cute.size(gW.layout[3].shape)):
        a=gW[tidx,None,bidx,kt].load().to(cutlass.Float32); b=gI[None,kt].load().to(cutlass.Float32)
        sa=gSFW[tidx,None,bidx,kt].load().to(cutlass.Float32); sb=gSFI[None,kt].load().to(cutlass.Float32)
        ra=cute.make_rmem_tensor_like(a);rb=cute.make_rmem_tensor_like(b);rsa=cute.make_rmem_tensor_like(sa);rsb=cute.make_rmem_tensor_like(sb)
        ra.store(a);rb.store(b);rsa.store(sa);rsb.store(sb)
        acc=frag[0]
        for i in cutlass.range_constexpr(MMA[2]): acc=acc+ra[i]*rsa[i]*rb[i]*rsb[i]
        frag[0]=acc
    h=bidx*128+tidx; contrib=frag[0]*rw
    cute.arch.atomic_add(mOut.iterator+(t*HIDDEN+h), contrib)

@cute.jit
def launch_dn(mW2,mI,mSFW2_raw,mSFI_raw,meidx,mtidx,mRW,mOut, E:cutlass.Constexpr,HIDDEN:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,L:cutlass.Constexpr, stream):
    MMA=(128,1,BK); ks=INTER//SVS
    mSFW2=cute.make_tensor(mSFW2_raw.iterator, cute.make_layout((E,HIDDEN,(SVS,ks)),stride=(HIDDEN*ks,ks,(0,1))))
    mSFI =cute.make_tensor(mSFI_raw.iterator, cute.make_layout((L,(SVS,ks)),stride=(ks,(0,1))))
    down_k(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,HIDDEN,MMA).launch(grid=(cute.ceil_div(HIDDEN,128),1,L),block=(128,1,1),stream=stream)

# ---------- NVFP4 quant / dequant helpers ----------
def quant_nvfp4(x, dev):  # x (R,Kk) fp32 -> packed (R,Kk//2) uint8, sf (R,Kk//16) uint8(e4m3)
    R,Kk=x.shape; nb=Kk//SVS
    xb=x.reshape(R,nb,SVS)
    amax=xb.abs().amax(-1).clamp(min=1e-6)
    scale_q=(amax/6.0).to(torch.float8_e4m3fn); scale_f=scale_q.float().clamp(min=1e-12)
    xn=(xb/scale_f.unsqueeze(-1))
    grid=torch.tensor([0,.5,1,1.5,2,3,4,6],device=dev)
    idx=(xn.abs().unsqueeze(-1)-grid).abs().argmin(-1)
    nib=(idx+(xn<0).long()*8).to(torch.uint8).reshape(R,Kk)
    packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8)
    return packed.contiguous(), scale_q.view(torch.uint8).reshape(R,nb).contiguous()

def deq2(p,s,kk,dev):
    lut=torch.tensor(E2,device=dev); lo=(p&0xF).long(); hi=((p>>4)&0xF).long(); R=p.shape[0]
    v=torch.empty(R,kk,device=dev); v[:,0::2]=lut[lo]; v[:,1::2]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(SVS,dim=1)

dev="cuda"; torch.manual_seed(2)
E,INTER,HIDDEN,T,L = 8,2048,7168,8,64; M2=2*INTER; ksH=HIDDEN//SVS; ksI=INTER//SVS
# FC1 weights (E,2*INTER,HIDDEN), x (T,HIDDEN), FC2 weights (E,HIDDEN,INTER)
Wu=torch.randint(0,256,(E,M2,HIDDEN//2),device=dev,dtype=torch.uint8); SFW=torch.randint(1,15,(E,M2,ksH),device=dev,dtype=torch.uint8).contiguous()
Xu=torch.randint(0,256,(T,HIDDEN//2),device=dev,dtype=torch.uint8); SFX=torch.randint(1,15,(T,ksH),device=dev,dtype=torch.uint8).contiguous()
W2u=torch.randint(0,256,(E,HIDDEN,INTER//2),device=dev,dtype=torch.uint8); SFW2=torch.randint(1,15,(E,HIDDEN,ksI),device=dev,dtype=torch.uint8).contiguous()
eidx=torch.randint(0,E,(L,),device=dev,dtype=torch.int32); tidx=torch.randint(0,T,(L,),device=dev,dtype=torch.int32); rw=torch.rand(L,device=dev,dtype=torch.float32)
Inter=torch.zeros(INTER,L,device=dev,dtype=torch.float32); Out=torch.zeros(T,HIDDEN,device=dev,dtype=torch.float32)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
# --- gate_up ---
mW=from_dlpack(Wu.view(torch.float4_e2m1fn_x2)); mX=from_dlpack(Xu.view(torch.float4_e2m1fn_x2))
mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn))
meidx=from_dlpack(eidx); mtidx=from_dlpack(tidx); mInter=from_dlpack(Inter)
cgu=cute.compile(launch_gu,mW,mX,mSFW,mSFX,meidx,mtidx,mInter,E,M2,HIDDEN,T,L,INTER,cur)
cgu(mW,mX,mSFW,mSFX,meidx,mtidx,mInter,cur); torch.cuda.synchronize()
# --- quantize intermediate to NVFP4 ---
Inter_LI=Inter.t().contiguous()                          # (L, INTER)
Iu_q, SFI_q = quant_nvfp4(Inter_LI, dev)                 # fp4 + e4m3 sf
inter_q = deq2(Iu_q, SFI_q, INTER, dev)                  # values FC2 actually sees
# --- down ---
mW2=from_dlpack(W2u.view(torch.float4_e2m1fn_x2)); mI=from_dlpack(Iu_q.view(torch.float4_e2m1fn_x2))
mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mSFI=from_dlpack(SFI_q.view(torch.float8_e4m3fn))
mRW=from_dlpack(rw); mOut=from_dlpack(Out)
cdn=cute.compile(launch_dn,mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,E,HIDDEN,INTER,T,L,cur)
cdn(mW2,mI,mSFW2,mSFI,meidx,mtidx,mRW,mOut,cur); torch.cuda.synchronize()

# ---------- references ----------
Wf=deq2(Wu.reshape(E*M2,HIDDEN//2),SFW.reshape(E*M2,ksH),HIDDEN,dev).reshape(E,M2,HIDDEN); Xf=deq2(Xu,SFX,HIDDEN,dev)
W2f=deq2(W2u.reshape(E*HIDDEN,INTER//2),SFW2.reshape(E*HIDDEN,ksI),INTER,dev).reshape(E,HIDDEN,INTER)
out_q=torch.zeros(T,HIDDEN,device=dev); out_fp=torch.zeros(T,HIDDEN,device=dev)
for l in range(L):
    g=Wf[eidx[l],:INTER]@Xf[tidx[l]]; u=Wf[eidx[l],INTER:]@Xf[tidx[l]]; itr=(g*torch.sigmoid(g))*u  # fp32 intermediate
    out_fp[tidx[l]] += (W2f[eidx[l]]@itr)*rw[l].item()
    out_q[tidx[l]]  += (W2f[eidx[l]]@inter_q[l])*rw[l].item()                                          # quantized intermediate
cosk=torch.nn.functional.cosine_similarity(Out.flatten().unsqueeze(0),out_q.flatten().unsqueeze(0)).item()
relk=(Out-out_q).norm().item()/(out_q.norm().item()+1e-9)
cosf=torch.nn.functional.cosine_similarity(Out.flatten().unsqueeze(0),out_fp.flatten().unsqueeze(0)).item()
print(f"FULL MoE  kernel-vs-quantref  cosine={cosk:.6f} relerr={relk:.6f} -> {'PASS' if cosk>0.999 and relk<0.02 else 'FAIL'}")
print(f"FULL MoE  kernel-vs-fp32ref   cosine={cosf:.6f} relerr={(Out-out_fp).norm().item()/(out_fp.norm().item()+1e-9):.6f}  (quant-error floor)")
print("Out[0,:3]",[round(x,3) for x in Out[0,:3].tolist()]," quantref",[round(x,3) for x in out_q[0,:3].tolist()])

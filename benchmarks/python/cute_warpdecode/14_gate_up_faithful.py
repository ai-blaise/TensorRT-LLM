import torch, cutlass, cutlass.cute as cute, operator
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
WARPS=8  # warps per CTA

@cute.kernel
def gu_k(mW, mX, mSFW, mSFX, mEids, mInter, NW: cutlass.Constexpr, INTER: cutlass.Constexpr, TOPK: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx()
    warp=tidx//32; lane=tidx%32
    j=bidx*WARPS+warp                      # intermediate neuron
    t=bidz//TOPK; ks=bidz%TOPK             # token, top-k slot
    e=mEids[bidz]                          # routed expert id for this (token,slot)
    gacc=cutlass.Float32(0.0); uacc=cutlass.Float32(0.0)
    w=lane
    while w < NW:
        xw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mX[t,w].ir_value()),(8,),cutlass.Float16)
        gw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e,j,w].ir_value()),(8,),cutlass.Float16)
        uw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW[e,j+INTER,w].ir_value()),(8,),cutlass.Float16)
        sfx=cutlass.Float32(mSFX[t,w//2]); sfg=cutlass.Float32(mSFW[e,j,w//2]); sfu=cutlass.Float32(mSFW[e,j+INTER,w//2])
        gp=(gw*xw).to(cutlass.Float32); upp=(uw*xw).to(cutlass.Float32)
        gs=cutlass.Float32(0.0); us=cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8): gs=gs+gp[i]; us=us+upp[i]
        gacc=gacc+gs*sfg*sfx; uacc=uacc+us*sfu*sfx
        w=w+32
    g=cute.arch.warp_reduction_sum(gacc); u=cute.arch.warp_reduction_sum(uacc)
    if lane==0:
        silu=g*(cutlass.Float32(1.0)/(cutlass.Float32(1.0)+cute.arch.exp(-g)))
        mInter[bidz,j]=(silu*u).to(cutlass.BFloat16)

@cute.jit
def launch(mW,mX,mSFW_raw,mSFX_raw,mEids,mInter, E:cutlass.Constexpr,M2:cutlass.Constexpr,H:cutlass.Constexpr,T:cutlass.Constexpr,LP:cutlass.Constexpr,INTER:cutlass.Constexpr,TOPK:cutlass.Constexpr, stream):
    NW=H//8; sfk=H//16
    mSFW=cute.make_tensor(mSFW_raw.iterator, cute.make_layout((E,M2,sfk),stride=(M2*sfk,sfk,1)))
    mSFX=cute.make_tensor(mSFX_raw.iterator, cute.make_layout((T,sfk),stride=(sfk,1)))
    gu_k(mW,mX,mSFW,mSFX,mEids,mInter,NW,INTER,TOPK).launch(grid=(cute.ceil_div(INTER,WARPS),1,LP),block=(WARPS*32,1,1),stream=stream)

dev="cuda"; torch.manual_seed(0)
E,INTER,H,T,TOPK = 16,2048,7168,32,8; M2=2*INTER; LP=T*TOPK; NW=H//8; sfk=H//16
Wu=torch.randint(0,2**31,(E,M2,NW),device=dev,dtype=torch.int32); SFW=torch.randint(1,15,(E,M2,sfk),device=dev,dtype=torch.uint8).contiguous()
Xu=torch.randint(0,2**31,(T,NW),device=dev,dtype=torch.int32); SFX=torch.randint(1,15,(T,sfk),device=dev,dtype=torch.uint8).contiguous()
eids=torch.randint(0,E,(LP,),device=dev,dtype=torch.int32)    # expert per (token,slot)
Inter=torch.zeros(LP,INTER,device=dev,dtype=torch.bfloat16)
mW=from_dlpack(Wu); mX=from_dlpack(Xu); mSFW=from_dlpack(SFW.view(torch.float8_e4m3fn)); mSFX=from_dlpack(SFX.view(torch.float8_e4m3fn))
mEids=from_dlpack(eids); mInter=from_dlpack(Inter)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
comp=cute.compile(launch,mW,mX,mSFW,mSFX,mEids,mInter,E,M2,H,T,LP,INTER,TOPK,cur)
comp(mW,mX,mSFW,mSFX,mEids,mInter,cur); torch.cuda.synchronize()
E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
def deq32(p,s,KK):
    R=p.shape[0]; pu=p.view(torch.uint8).view(R,KK//8,4); v=torch.empty(R,KK,device=dev)
    for b in range(4):
        lo=(pu[:,:,b]&0xF).long(); hi=((pu[:,:,b]>>4)&0xF).long()
        v[:,(b*2)::8]=lut[lo]; v[:,(b*2+1)::8]=lut[hi]
    return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
Wf=deq32(Wu.reshape(E*M2,NW),SFW.reshape(E*M2,sfk),H).reshape(E,M2,H); Xf=deq32(Xu,SFX,H)
ref=torch.empty(LP,INTER,device=dev)
for l in range(LP):
    t=l//TOPK; e=eids[l]
    g=Wf[e,:INTER]@Xf[t]; u=Wf[e,INTER:]@Xf[t]; ref[l]=((g*torch.sigmoid(g))*u).to(torch.bfloat16)
cos=torch.nn.functional.cosine_similarity(Inter.float().flatten().unsqueeze(0),ref.float().flatten().unsqueeze(0)).item()
print(f"GU-FAITHFUL cosine={cos:.6f} -> {'PASS' if cos>0.999 else 'FAIL'}")
def bench(fn,it=50,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
t=bench(lambda: comp(mW,mX,mSFW,mSFX,mEids,mInter,cur)); wb=LP*(M2*H//2)
print(f"GU-FAITHFUL E={E} LP={LP}: {t:.2f}us  BW={wb/(t*1e-6)/1e12:.2f}TB/s")

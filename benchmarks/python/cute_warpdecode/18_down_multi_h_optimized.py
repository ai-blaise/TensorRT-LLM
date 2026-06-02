import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
WARPS=8; HPW=8   # output-h handled per warp (amortize inter read, wider w2 coalescing)

@cute.kernel
def dn_k(mW2, mI3, mSFW2, mEids, mRW, mOut, NW: cutlass.Constexpr, TOPK: cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx(); warp=tidx//32; lane=tidx%32
    h0=(bidx*WARPS+warp)*HPW; t=bidz
    accs=cute.make_fragment(HPW, cutlass.Float32)
    for q in cutlass.range_constexpr(HPW): accs[q]=cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(TOPK):
        pair=t*TOPK+ks; e=mEids[pair]; rw=cutlass.Float32(mRW[pair]); w=lane
        parts=cute.make_fragment(HPW, cutlass.Float32)
        for q in cutlass.range_constexpr(HPW): parts[q]=cutlass.Float32(0.0)
        while w<NW:
            iv=mI3[pair,w,None].load().to(cutlass.Float16)        # read inter ONCE, reuse for all HPW
            sfb=cutlass.Float32(1.0)  # inter scale folded in gate_up output (BF16, scale=1)
            for q in cutlass.range_constexpr(HPW):
                dw=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(mW2[e,h0+q,w].ir_value()),(8,),cutlass.Float16)
                sfd=cutlass.Float32(mSFW2[e,h0+q,w//2])
                prod=(dw*iv).to(cutlass.Float32); s=cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(8): s=s+prod[i]
                parts[q]=parts[q]+s*sfd
            w=w+32
        for q in cutlass.range_constexpr(HPW): accs[q]=accs[q]+parts[q]*rw
    for q in cutlass.range_constexpr(HPW):
        o=cute.arch.warp_reduction_sum(accs[q])
        if lane==0: mOut[t,h0+q]=o.to(cutlass.BFloat16)

@cute.jit
def launch(mW2,mI3,mSFW2_raw,mEids,mRW,mOut, E:cutlass.Constexpr,H:cutlass.Constexpr,INTER:cutlass.Constexpr,T:cutlass.Constexpr,LP:cutlass.Constexpr,TOPK:cutlass.Constexpr, stream):
    NW=INTER//8; sfk=INTER//16
    mSFW2=cute.make_tensor(mSFW2_raw.iterator, cute.make_layout((E,H,sfk),stride=(H*sfk,sfk,1)))
    dn_k(mW2,mI3,mSFW2,mEids,mRW,mOut,NW,TOPK).launch(grid=(cute.ceil_div(H,WARPS*HPW),1,T),block=(WARPS*32,1,1),stream=stream)

dev="cuda"; torch.manual_seed(1)
def run(E,H,INTER,T,TOPK,label):
    LP=T*TOPK; NW=INTER//8; sfk=INTER//16
    W2u=torch.randint(0,2**31,(E,H,NW),device=dev,dtype=torch.int32); SFW2=torch.randint(1,15,(E,H,sfk),device=dev,dtype=torch.uint8).contiguous()
    Inter=(torch.randn(LP,INTER,device=dev,dtype=torch.bfloat16)*0.1); Inter3=Inter.view(LP,NW,8)
    eids=torch.randint(0,E,(LP,),device=dev,dtype=torch.int32); rw=torch.rand(LP,device=dev,dtype=torch.float32)
    Out=torch.zeros(T,H,device=dev,dtype=torch.bfloat16)
    mW2=from_dlpack(W2u); mI3=from_dlpack(Inter3); mSFW2=from_dlpack(SFW2.view(torch.float8_e4m3fn)); mEids=from_dlpack(eids); mRW=from_dlpack(rw); mOut=from_dlpack(Out)
    cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    comp=cute.compile(launch,mW2,mI3,mSFW2,mEids,mRW,mOut,E,H,INTER,T,LP,TOPK,cur)
    comp(mW2,mI3,mSFW2,mEids,mRW,mOut,cur); torch.cuda.synchronize()
    E2=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]; lut=torch.tensor(E2,device=dev)
    def deq(p,s,KK):
        R=p.shape[0]; pu=p.view(torch.uint8).view(R,KK//8,4); v=torch.empty(R,KK,device=dev)
        for b in range(4):
            lo=(pu[:,:,b]&0xF).long(); hi=((pu[:,:,b]>>4)&0xF).long(); v[:,(b*2)::8]=lut[lo]; v[:,(b*2+1)::8]=lut[hi]
        return v*s.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
    W2f=deq(W2u.reshape(E*H,NW),SFW2.reshape(E*H,sfk),INTER).reshape(E,H,INTER); If=Inter.float()
    ref=torch.zeros(T,H,device=dev)
    for l in range(LP): t=l//TOPK; e=eids[l]; ref[t]+=(W2f[e]@If[l])*rw[l].item()
    cos=torch.nn.functional.cosine_similarity(Out.float().flatten().unsqueeze(0),ref.flatten().unsqueeze(0)).item()
    def bench(fn,it=50,wu=10):
        for _ in range(wu): fn()
        torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
    tt=bench(lambda: comp(mW2,mI3,mSFW2,mEids,mRW,mOut,cur)); wb=LP*(H*INTER//2)
    print(f"{label}: cos={cos:.5f} {tt:.1f}us weight-BW={wb/(tt*1e-6)/1e12:.2f}TB/s (HPW={HPW})")
run(16,7168,2048,32,8,"DN-v3 E16 LP256")
run(16,7168,2048,2,8,"DN-v3 E16 LP16(no-share)")

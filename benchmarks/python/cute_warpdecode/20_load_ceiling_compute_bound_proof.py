import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
import cuda.bindings.driver as cuda
WARPS=8
# output-owned weight-load ceiling: load uint32 weight rows (no cvt, dummy accumulate) at gate_up pattern
@cute.kernel
def k(mW,mEids,mOut,NWk:cutlass.Constexpr,MODE:cutlass.Constexpr):
    bidx,_,bidz=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx(); warp=tidx//32; lane=tidx%32
    j=bidx*WARPS+warp; e=mEids[bidz]; acc=cutlass.Int32(0); w=lane
    while w<NWk:
        word=mW[e,j,w]
        if MODE==0:    # pure load (xor accumulate raw bits) -> LDG ceiling
            acc=acc^cutlass.Int32(word)
        else:          # load + cvt to f16 (no FFMA beyond reduce) -> cvt cost
            f=TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(word.ir_value()),(8,),cutlass.Float16)
            acc=acc^cutlass.Int32(cutlass.Float32(f[0]))
        w=w+32
    r=cute.arch.warp_reduction_sum(cutlass.Float32(acc))
    if lane==0: mOut[bidz,j]=r
@cute.jit
def launch(mW,mEids,mOut,M2:cutlass.Constexpr,H:cutlass.Constexpr,LP:cutlass.Constexpr,MODE:cutlass.Constexpr,stream):
    NWk=H//8
    k(mW,mEids,mOut,NWk,MODE).launch(grid=(cute.ceil_div(M2,WARPS),1,LP),block=(WARPS*32,1,1),stream=stream)
dev="cuda"; torch.manual_seed(0)
E,INTER,H,LP=16,2048,7168,16; M2=2*INTER; NWh=H//8
Wu=torch.randint(0,2**31,(E,M2,NWh),device=dev,dtype=torch.int32); eids=torch.randint(0,E,(LP,),device=dev,dtype=torch.int32)
Out=torch.zeros(LP,M2,device=dev,dtype=torch.float32)
mW=from_dlpack(Wu); mEids=from_dlpack(eids); mOut=from_dlpack(Out)
cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def bench(comp,it=50,wu=10):
    f=lambda: comp(mW,mEids,mOut,cur)
    for _ in range(wu): f()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
wb=LP*(M2*H//2)  # bytes of the gate weight (M2 rows = gate+up) read per pair
for MODE in [0,1]:
    comp=cute.compile(launch,mW,mEids,mOut,M2,H,LP,MODE,cur)
    t=bench(comp)
    print(f"MODE={'load-only' if MODE==0 else 'load+cvt'}: {t:.1f}us  {wb/(t*1e-6)/1e12:.2f}TB/s")

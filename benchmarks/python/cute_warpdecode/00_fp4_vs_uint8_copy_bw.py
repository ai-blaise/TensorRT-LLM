import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

# Pure bandwidth test: copy a big buffer gmem->gmem in CuTeDSL, three dtypes.
def make_kernel(VEC):
    @cute.kernel
    def copy_k(src, dst, NTILE: cutlass.Constexpr):
        bidx,_,_=cute.arch.block_idx(); tidx,_,_=cute.arch.thread_idx()
        gi = (bidx*256 + tidx)
        gs=cute.local_tile(src,(VEC,),(None,)); gd=cute.local_tile(dst,(VEC,),(None,))
        f=cute.make_rmem_tensor_like(gs[None,gi]); cute.autovec_copy(gs[None,gi], f); cute.autovec_copy(f, gd[None,gi])
    return copy_k

@cute.jit
def launch(src,dst, N:cutlass.Constexpr, VEC:cutlass.Constexpr, stream):
    k=make_kernel(VEC)
    k(src,dst,N//VEC).launch(grid=(cute.ceil_div(N//VEC,256),1,1),block=(256,1,1),stream=stream)

dev="cuda"
def run(label, dtype, view_fp4=False, VEC=128):
    N=1<<26  # 64M elements
    if view_fp4:
        su=torch.randint(0,256,(N//2,),device=dev,dtype=torch.uint8); du=torch.zeros(N//2,device=dev,dtype=torch.uint8)
        src=from_dlpack(su.view(torch.float4_e2m1fn_x2)); dst=from_dlpack(du.view(torch.float4_e2m1fn_x2)); bytes_moved=2*(N//2)
    else:
        s=torch.randn(N,device=dev,dtype=dtype); d=torch.zeros(N,device=dev,dtype=dtype)
        src=from_dlpack(s); dst=from_dlpack(d); bytes_moved=2*N*s.element_size()
    cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    comp=cute.compile(launch,src,dst,N,VEC,cur)
    for _ in range(5): comp(src,dst,cur)
    torch.cuda.synchronize(); ev1=torch.cuda.Event(True); ev2=torch.cuda.Event(True); ev1.record()
    for _ in range(30): comp(src,dst,cur)
    ev2.record(); torch.cuda.synchronize(); t=ev1.elapsed_time(ev2)/30*1000
    print(f"{label:20s} VEC={VEC:3d}: {t:.2f}us  BW(r+w)={bytes_moved/(t*1e-6)/1e12:.2f}TB/s")

def run_u8(label, VEC):
    N=1<<25  # bytes (= same byte volume as fp4 N=1<<26 elements)
    s=torch.randint(0,256,(N,),device=dev,dtype=torch.uint8); d=torch.zeros(N,device=dev,dtype=torch.uint8)
    src=from_dlpack(s); dst=from_dlpack(d); bytes_moved=2*N
    cur=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    comp=cute.compile(launch,src,dst,N,VEC,cur)
    for _ in range(5): comp(src,dst,cur)
    torch.cuda.synchronize(); ev1=torch.cuda.Event(True); ev2=torch.cuda.Event(True); ev1.record()
    for _ in range(30): comp(src,dst,cur)
    ev2.record(); torch.cuda.synchronize(); t=ev1.elapsed_time(ev2)/30*1000
    print(f"{label:20s} VEC={VEC:3d}: {t:.2f}us  BW(r+w)={bytes_moved/(t*1e-6)/1e12:.2f}TB/s")

run("fp32 copy", torch.float32, VEC=4)
run("fp4 copy VEC=128", torch.uint8, view_fp4=True, VEC=128)
run_u8("uint8 copy VEC=16", 16)   # 16 bytes/thread, same bytes as fp4
run_u8("uint8 copy VEC=4", 4)

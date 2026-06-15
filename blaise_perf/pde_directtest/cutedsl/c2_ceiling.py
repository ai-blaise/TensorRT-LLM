"""Establish the floor I'm racing: (1) pure HBM-read time for W at each shape (BW ceiling),
(2) empty CUDA-graph replay overhead, (3) a single bf16 copy of W-sized data,
(4) does forcing the stock cutedsl to higher cluster counts help? Probe nvfp4_gemm AutoTuner choice.
"""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return fp4,sf
def cap_us(fn,windows=8,it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); best=float("inf")
    for _ in range(windows):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best

# empty graph overhead
def empty(): return torch.empty(1,device=dev)
print("empty_graph_replay_us", round(cap_us(empty),3), flush=True)

shapes=[("o_proj",16384,7168),("q_a",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
print("BW floor: read W(fp4 = N*K/2 bytes) once + write C(M*N*2). B200 HBM ~8TB/s",flush=True)
for name,K,N in shapes:
    wbytes=N*K//2  # fp4 packed
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
    print(f"{name}: W_fp4_bytes={wbytes/1e6:.1f}MB  ideal_read@8TB/s={wbytes/8e12*1e6:.2f}us  wf.numel={wf.numel()}",flush=True)
    # pure read: sum the fp4 bytes (reinterpret as uint8) -> forces HBM read of W
    wf_u8=wf.view(torch.uint8)
    def readW(): return wf_u8.sum(dtype=torch.int64)
    print(f"  {name} pure_W_read_us", round(cap_us(readW),3), flush=True)
    # which backend does AutoTuner pick? compare default_auto vs cublaslt vs cutedsl at M=1
    M=1; x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1)
    for bn,backend in [("auto","cutlass,cublaslt,cutedsl,cuda_core"),("cublaslt","cublaslt"),("cutedsl_op","cutedsl")]:
        def fn(): return torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,backend,None)
        try: t=round(cap_us(fn),2)
        except Exception as e: t=f"ERR:{str(e)[:30]}"
        print(f"  {name} M1 {bn} {t}us",flush=True)
print("DONE",flush=True)

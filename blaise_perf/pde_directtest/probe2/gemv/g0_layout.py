"""g0: characterize NVFP4 fp4_quantize output layout for the custom GEMV kernel.
   vec=16 => swizzled MUST be False (plain). vec=32 ue8m0 => swizzled (tensor-core tiled)."""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0)

def q16(x):  # production dense path
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),16,False)
    return fp4,sf,g

def q32(x):  # ue8m0 swizzled (task-brief stated path)
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),32,True)
    return fp4,sf,g

def main():
    print(f"dev={torch.cuda.get_device_name(0)} torch={torch.__version__}",flush=True)
    for (name,K,N) in [("moe_up",7168,2048),("q_a",7168,1536),("o_proj",16384,7168),("q_b",1536,24576),("moe_down",2048,7168)]:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
        wf,wsf,wg=q16(w)
        print(f"{name} K={K} N={N} vec16: wf={tuple(wf.shape)}/{wf.dtype} (K/2={K//2}) wsf.numel={wsf.numel()} (N*K/16={N*K//16})",flush=True)
        try:
            wf2,wsf2,_=q32(w)
            print(f"   vec32-sw: wf={tuple(wf2.shape)}/{wf2.dtype} wsf.numel={wsf2.numel()} (N*K/32={N*K//32})",flush=True)
        except Exception as ex:
            print(f"   vec32-sw ERR {type(ex).__name__}:{str(ex)[:70]}",flush=True)
    print("--- activation SF layout (vec16) ---",flush=True)
    for M in (1,4,8):
        x=torch.randn(M,7168,device=dev,dtype=torch.bfloat16); xf,xsf,xg=q16(x)
        print(f"M={M} K=7168: xf={tuple(xf.shape)} xsf.numel={xsf.numel()} (M*K/16={M*7168//16}, ceil(M/128)*128*K/16={((M+127)//128)*128*7168//16})",flush=True)
    # empty-graph replay floor
    def noop(): return torch.empty(1,device=dev)
    for _ in range(8): noop()
    torch.cuda.synchronize()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): noop()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): noop()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    best=float("inf")
    for _ in range(5):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(200): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/200*1000.0)
    print(f"empty-graph replay floor: {best:.3f} us/replay",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()

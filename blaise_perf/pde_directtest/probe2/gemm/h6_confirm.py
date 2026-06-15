"""h6: confirm the cutedsl capture win is real, not a timing/fallback artifact.
  (1) Independent cross-check timing: longer replay (200), wall-clock vs cuda-event, on
      q_a_proj M=1 and o_proj M=16, for cutlass vs cutedsl vs default_auto.
  (2) Confirm cutedsl is actually IN the graph: capture a graph that runs cutedsl 10x;
      its per-replay time should be ~10x the single-call time (proves real work captured).
  (3) Explain dispatcher: time what default_auto SELECTS by checking its captured per-call
      time == which forced backend. Also run autotune() context to see if selection changes.
  (4) nsys-free occupancy sanity: compare cutedsl single-call captured time to a trivial
      empty-graph replay floor (to rule out 'cutedsl replay is just measuring nothing').
"""
import torch, tensorrt_llm, time
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import autotune
dev="cuda"; torch.manual_seed(0); SVS=16

def cq(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False)
    return fp4,sf,g

def cap_build(fn,reps=1):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(reps): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g):
        for _ in range(reps): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    return g

def event_us(g,it=200):
    a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it*1000.0

def wall_us(g,it=200):
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e6

def one(name,K,N,M):
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg=cq(w)
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg=cq(x); alpha=(1.0/(wg*xg)).reshape(1)
    cutlass=lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(xf,wf,xsf,wsf,alpha,torch.bfloat16)
    cutedsl=lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf,wf,xsf,wsf,alpha,torch.bfloat16)
    dauto  =lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cutlass,cublaslt,cutedsl,cuda_core",None)
    print(f"\n[{name} K={K} N={N} M={M}]",flush=True)
    for bn,fn in [("cutlass",cutlass),("cutedsl",cutedsl),("default_auto",dauto)]:
        g1=cap_build(fn,1); g10=cap_build(fn,10)
        e1=event_us(g1); w1=wall_us(g1); e10=event_us(g10)
        print(f"  {bn:13s}: 1x event={e1:7.2f}us wall={w1:7.2f}us | 10x event/10={e10/10:7.2f}us  (linear? {e10/10/e1:.2f}x)",flush=True)

def main():
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    # empty-graph floor
    def noop():
        return torch.empty(1,device=dev)
    gf=cap_build(noop,1); print(f"empty-graph replay floor: event={event_us(gf):.3f}us wall={wall_us(gf):.3f}us",flush=True)
    one("q_a_proj",7168,1536,1)
    one("o_proj",16384,7168,16)
    one("moe_down",2048,7168,16)
    # dispatcher selection under autotune() ctx (the prod path warms tuner here)
    print("\n=== dispatcher selection w/ autotune() ctx (q_a_proj M=16) ===",flush=True)
    K,N,M=7168,1536,16
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg=cq(w)
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg=cq(x); alpha=(1.0/(wg*xg)).reshape(1)
    def dauto(): return torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cutlass,cublaslt,cutedsl,cuda_core",None)
    with autotune():
        for _ in range(3): dauto()
    torch.cuda.synchronize()
    g=cap_build(dauto,1); print(f"  default_auto after autotune(): captured 1x = {event_us(g):.2f}us (cutedsl~10.3 cutlass~18.5 => which won?)",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()

"""h5: DEFINITIVE under-capture benchmark, CORRECT vec=16 NVFP4 layout + proper alpha.

For each dense GEMM shape x M in {1,4,16,64}:
  - quantize w,x at vec=16 swizzled=False; alpha = 1/(w_sf_global * x_sf_global)
  - correctness gate: cos vs torch.ops.trtllm.fp4_gemm reference (oracle); flag if < 0.999
  - EAGER us (100 iters) and CAPTURED us (min of 5 windows x 50 replays) for each backend
Backends: cutlass, cutedsl, cuda_core(M<=8), default_auto (cutlass,cublaslt,cutedsl,cuda_core),
          default_cublaslt (production default; will ERR in this image -> noted),
          default_prod_set (cutlass,cublaslt,cuda_core == prior-probe set).
Emits 'ROW <shape> <M> <backend> <eager_us> <cap_us> <ratio> <cos> <note>'.
"""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0)
SVS=16

def cq(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False)
    return fp4,sf,g

def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm()
    return (a@b/n).item() if n>0 else float('nan')

def eager_us(fn,it=100,wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000.0

def cap_us(fn,windows=5,it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g):
        h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    best=float("inf")
    for _ in range(windows):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best, h["o"].clone()

shapes=[("o_proj",16384,7168),("q_a_proj",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]

def main():
    print(f"dev={torch.cuda.get_device_name(0)} vec={SVS} (production dense NVFP4 layout)",flush=True)
    print("ROW shape M backend eager_us cap_us cap_vs_eager cos note",flush=True)
    for name,K,N in shapes:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg=cq(w)
        for M in (1,4,16,64):
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg=cq(x)
            alpha=(1.0/(wg*xg)).reshape(1)
            ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
            torch.cuda.synchronize()
            bks={
              "cutlass":  lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(xf,wf,xsf,wsf,alpha,torch.bfloat16),
              "cutedsl":  lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf,wf,xsf,wsf,alpha,torch.bfloat16),
              "default_auto":     lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cutlass,cublaslt,cutedsl,cuda_core",None),
              "default_prodset":  lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cutlass,cublaslt,cuda_core",None),
              "default_cublaslt": lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None),
            }
            if M<=8:
                bks["cuda_core"]=lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cuda_core",None)
            for bn,fn in bks.items():
                e=float("nan"); c=float("nan"); cs=float("nan"); note="ok"
                try:
                    y=fn().clone(); torch.cuda.synchronize(); cs=cos(y,ref)
                    if cs<0.999: note=f"LOWCOS"
                except Exception as ex:
                    note=f"ERR:{type(ex).__name__}:{str(ex)[:50]}"
                    print(f"ROW {name} {M} {bn} nan nan nan nan {note}",flush=True); continue
                try: e=eager_us(fn)
                except Exception as ex: note=f"eager_{type(ex).__name__}"
                try: c,_=cap_us(fn)
                except Exception as ex: note=f"cap_{type(ex).__name__}:{str(ex)[:40]}"
                r=(e/c) if (c==c and c>0) else float("nan")
                print(f"ROW {name} {M} {bn} {e:.2f} {c:.2f} {r:.2f} {cs:.5f} {note}",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()

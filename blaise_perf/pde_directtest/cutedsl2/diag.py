"""R2 diag: instantiate the production kernel for each shape@M=1, print computed stages + occupancy,
and the achieved DRAM BW of the round-1 winner (to see headroom toward the 8TB/s W-read floor).
This determines whether the kernel is already pipeline-saturated (split-K useless) or has stage headroom."""
import torch, tensorrt_llm, cutlass
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel as K
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    f,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return f,sf
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm(); return (a@b/n).item()
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

print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
# Inspect computed stages for the WIN tactic config per shape.
# WIN tactic = mma(256,64), cluster(4,1), swap_ab=True -> kernel_m=N, kernel_n=M
configs=[("o_proj",16384,7168),("q_a",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
WIN_mma=(256,64); WIN_clu=(4,1)
for name,K_,N in configs:
    try:
        kern=K(SVS, WIN_mma, WIN_clu, False)
        print(f"{name:9} WIN_mma={WIN_mma} clu={WIN_clu}: occupancy={kern.occupancy} "
              f"use_2cta={kern.use_2cta_instrs} (stages computed at __call__ time)",flush=True)
    except Exception as e:
        print(f"{name}: ctor-only {type(e).__name__}: {str(e)[:80]}",flush=True)

# Achieved BW of winner: bytes read = W_fp4 (N*K/2) + SF + X. W dominates.
print("\n--- achieved BW at WIN vs 8TB/s W-read floor ---",flush=True)
runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
WIN=((256,64),(4,1),True,False)
for name,K_,N in configs:
    w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
    Wbytes=N*K_//2  # fp4 packed
    SFbytes=wf.numel()*0  # approx; SF is N*K/16 bytes fp8
    SFbytes=N*(K_//16)
    for M in (1,64):
        x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
        alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
        t=cap_us(lambda: runner.forward(inputs,tactic=WIN))
        total_bytes=Wbytes+SFbytes+M*K_//2
        bw=total_bytes/(t*1e-6)/1e12  # TB/s
        floor=Wbytes/8e12*1e6  # us at 8TB/s
        print(f"{name:9} M={M:>2} t={t:.2f}us  W={Wbytes/1e6:.1f}MB SF={SFbytes/1e6:.2f}MB  "
              f"achieved_BW={bw:.2f}TB/s  W-read-floor={floor:.2f}us  off={t/floor:.2f}x",flush=True)
print("DONE",flush=True)

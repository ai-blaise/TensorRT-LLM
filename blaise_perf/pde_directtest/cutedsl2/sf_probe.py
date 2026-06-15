"""R2 SF-staging probe: is o_proj's ~14.3us wall the nvf4 (vec=16) SF->TMEM serialization?
Compare on identical o_proj shape: nvf4 (sf_vec=16, SF 4x heavy) vs mxf4 (sf_vec=32, SF light, ue8m0 SF).
If mxf4 is much faster, SF-staging IS the wall and a custom overlapped-SF mainloop is the lever.
Also test: does the dense_blockscaled_gemm_swiglu_fusion variant (fused epilogue) change o_proj timing?"""
import torch, tensorrt_llm, itertools, cutlass
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel as K
dev="cuda"; torch.manual_seed(0)
def gscale(x,sv): return (448.0*6.0)/x.abs().max().float() if sv==16 else (448.0)/x.abs().max().float()
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
name,K_,N="o_proj",16384,7168; M=1
w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16)
runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
WIN=((256,64),(4,1),True,False)
# nvf4 (vec16) - production
for sv in (16,32):
    wg=gscale(w,sv); xg=gscale(x,sv)
    wf,wsf=torch.ops.trtllm.fp4_quantize(w,wg.reshape(1),sv,False)
    xf,xsf=torch.ops.trtllm.fp4_quantize(x,xg.reshape(1),sv,False)
    alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
    # runner is constructed with sf_vec via env? check forward uses sf_vec_size const. Try tactic.
    try:
        # build kernel directly to control sf_vec
        kern=K(sv, WIN[0], WIN[1], WIN[3])
        t=cap_us(lambda: runner.forward(inputs,tactic=WIN))
        # cos vs fp4_gemm only valid for vec16 (oracle is W4A4_NVFP4)
        note=""
        if sv==16:
            ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
            y=runner.forward(inputs,tactic=WIN).clone(); torch.cuda.synchronize(); note=f"cos={cos(y,ref):.5f}"
        print(f"o_proj M=1 sf_vec={sv:>2} (SF {'heavy 4x' if sv==16 else 'light 1x'}): {t:.2f}us {note}",flush=True)
    except Exception as e:
        print(f"sf_vec={sv}: {type(e).__name__}: {str(e)[:90]}",flush=True)
print("DONE",flush=True)

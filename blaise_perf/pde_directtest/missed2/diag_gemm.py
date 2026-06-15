"""Diagnose which backend runs for q_b / indexer shapes and whether output is finite."""
import torch, tensorrt_llm
dev = "cuda"; torch.manual_seed(0)
qz = torch.ops.trtllm.tunable_fp4_quantize

def quant(x, sw=True):
    gs = (x.abs().max().float()/(448.0*6.0)).clamp_min(1e-6).reshape(1)
    o = qz(x, gs, 32, sw); return o[0], o[1], gs

shapes = [("q_b", 1536, 24576), ("idx_wqb", 1536, 8192), ("idx_wkwp", 7168, 192)]
for be in ["cutlass", "cuda_core", "cutlass,cuda_core", "cutedsl"]:
    print(f"--- backend={be} ---")
    for nm, K, N in shapes:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16); wf, wsf, wgs = quant(w)
        for M in [1, 64]:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16); af, asf, ags = quant(a)
            al = (ags*wgs).reshape(1)
            try:
                if be == "cutedsl":
                    out = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(af, wf, asf, wsf, al, torch.bfloat16)
                else:
                    out = torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, be, None)
                torch.cuda.synchronize()
                fin = torch.isfinite(out).all().item()
                print(f"  {nm:10} M={M:3} shape={tuple(out.shape)} finite={fin} max={out.abs().max().item():.3f}")
            except Exception as e:
                print(f"  {nm:10} M={M:3} ERR {str(e)[:50]}")

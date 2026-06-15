import torch, tensorrt_llm
allops = sorted(dir(torch.ops.trtllm))
print("TOTAL trtllm ops:", len(allops))
cats = {
  "nvfp4/fp4 gemm": ["fp4","nvfp4"],
  "fp8 gemm": ["fp8"],
  "generic gemm/bmm": ["gemm","bmm","matmul","_mm"],
  "moe/expert": ["moe","expert","warp"],
  "quant/dequant/scale": ["quant","dequant","scale"],
  "mla/attention": ["mla","attn","mqa","flash","rope"],
  "comm": ["allreduce","all_reduce","alltoall","all_to_all","allgather","reduce_scatter","ar_"],
  "kv/cache": ["kvarn","_cache","hisa","kv_"],
}
seen=set()
for cat,kws in cats.items():
    ops=[o for o in allops if any(k in o.lower() for k in kws)]
    print(f"\n=== {cat} ({len(ops)}) ===")
    for o in ops:
        seen.add(o); print("  ",o)
# schemas for the GEMM + MoE candidates (decode compute bulk)
print("\n\n##### SCHEMAS (gemm/moe/quant) #####")
for o in sorted(seen):
    if any(k in o.lower() for k in ["fp4","nvfp4","moe","expert","warp","gemm","bmm"]):
        op=getattr(torch.ops.trtllm,o,None)
        try: print(f"{o} :: {str(op.default._schema)}")
        except Exception: pass

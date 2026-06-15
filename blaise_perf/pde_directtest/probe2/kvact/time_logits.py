# CUDA-graph timing of CuTe DSL FP4 vs FP8 paged MQA logits (indexer KV-scan)
# at prod decode shapes. Reuses the standalone harness input builders.
import sys, time, statistics
sys.argv = ["probe"]
HD = "/host_repo/tests/scripts/cute_dsl_kernels/paged_mqa_logits"
sys.path.insert(0, HD)
sys.path.insert(0, "/host_repo/tensorrt_llm/_torch/cute_dsl_kernels")
import torch, cutlass
import run_fp4 as F4
import run_fp8 as F8

dev="cuda"
NUM_SMS=148
PBK=64           # tokens_per_block from yaml
H=64; D=128      # index_n_heads, index_head_dim

def build_inputs(B, ctx, max_ctx):
    torch.manual_seed(0)
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    n_blk = (context_lens + PBK - 1)//PBK
    total = int(n_blk.sum().item())
    num_blocks = total + B*2
    max_blk = int(n_blk.max().item())
    block_table = torch.zeros((B,max_blk), dtype=torch.int32, device=dev)
    pool = torch.randperm(num_blocks, device=dev, dtype=torch.int32)
    off=0
    for i,nb in enumerate(n_blk.tolist()):
        block_table[i,:nb]=pool[off:off+nb]; off+=nb
    q = torch.randn((B,1,H,D), device=dev, dtype=torch.bfloat16)
    kv = torch.randn((num_blocks,PBK,1,D), device=dev, dtype=torch.bfloat16)
    weights = torch.randn((B,H), device=dev, dtype=torch.float32)
    sched = F4._compute_schedule_metadata(context_lens.cpu(),128,NUM_SMS).to(dev)
    return context_lens, block_table, q, kv, weights, sched, num_blocks

def time_graph(fn, iters=50, warmup=10):
    # warmup + capture
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    # time replays
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    ts=[]
    for _ in range(iters):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000.0) # us
    return statistics.median(ts), min(ts)

for ctx, max_ctx in [(4608,132096),(16384,132096),(65536,132096)]:
    cl,bt,q,kv,w,sched,nb = build_inputs(1, ctx, max_ctx)
    # FP4 inputs
    qp, sfqp = F4._per_token_cast_to_fp4(q.view(-1,D), gran_k=32)
    q_fp4 = qp.view(torch.uint8).view(1,1,H,D//2)
    sf_q = sfqp.view(torch.int32).view(1,1,H)
    kv_f4,_ = F4._kv_cache_cast_to_fp4(kv)
    def run4():
        return F4.fp4_paged_mqa_logits(q_fp4,sf_q,kv_f4,w,cl,bt,sched,max_ctx,
            num_epi_subtiles=1, epi_dtype=cutlass.Float32, output_dtype=cutlass.Float16, num_sms=NUM_SMS)
    # FP8 inputs
    q_f8 = q.view(-1,D).to(torch.float8_e4m3fn).view(1,1,H,D)
    kv_f8,_ = F8._kv_cache_cast_to_fp8(kv) if hasattr(F8,"_kv_cache_cast_to_fp8") else (None,None)
    have_f8 = kv_f8 is not None
    def run8():
        return F8.fp8_paged_mqa_logits(q_f8,kv_f8,w,cl,bt,sched,max_ctx,
            num_epi_subtiles=1, epi_dtype=torch.float32, acc_dtype=torch.float32, output_dtype=torch.float16, num_sms=NUM_SMS)
    try:
        m4,mn4 = time_graph(run4)
        # bytes read for KV (fp4): num K tokens scanned ~ ctx; per token D//2+4 bytes
        kv_bytes_f4 = ctx*(D//2+4)
        print(f"[FP4] ctx={ctx} width={max_ctx}: median={m4:.2f}us min={mn4:.2f}us  KVbytes~{kv_bytes_f4/1024:.1f}KB", flush=True)
    except Exception as ex:
        print(f"[FP4] ctx={ctx} ERR {repr(ex)[:300]}", flush=True)
    if have_f8:
        try:
            m8,mn8 = time_graph(run8)
            kv_bytes_f8 = ctx*(D+4)
            print(f"[FP8] ctx={ctx} width={max_ctx}: median={m8:.2f}us min={mn8:.2f}us  KVbytes~{kv_bytes_f8/1024:.1f}KB", flush=True)
        except Exception as ex:
            print(f"[FP8] ctx={ctx} ERR {repr(ex)[:300]}", flush=True)
    else:
        print(f"[FP8] no _kv_cache_cast_to_fp8 in run_fp8; will inline next iter", flush=True)

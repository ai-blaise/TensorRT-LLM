import torch, tensorrt_llm  # noqa
dev = 'cuda'
torch.manual_seed(0)

def bench(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

op = torch.ops.trtllm.cute_dsl_indexer_topk_decode

def run_case(B, num_cols, top_k, dt):
    logits = torch.randn(B, num_cols, dtype=dt, device=dev)
    seq_lens = torch.full((B,), num_cols, dtype=torch.int32, device=dev)
    out_idx = torch.full((B, top_k), -1, dtype=torch.int32, device=dev)
    def call(): op(logits, seq_lens, out_idx, top_k)
    call(); torch.cuda.synchronize()
    gold = torch.topk(logits.float(), min(top_k, num_cols), dim=1).indices
    ok = 0
    for b in range(B):
        a = set(int(x) for x in out_idx[b].tolist() if x >= 0)
        g = set(int(x) for x in gold[b].tolist())
        inter = len(a & g)
        if inter >= min(top_k, num_cols) * 0.99: ok += 1
    t = bench(call)
    print(f"B={B:>2} cols={num_cols:>6} topk={top_k:>4} {str(dt):>14}: {t*1000:8.1f} us  setmatch={ok}/{B}")

print("=== cute_dsl_indexer_topk_decode baseline (prod two-level shapes) ===")
for dt in (torch.float32, torch.bfloat16):
    for (B, cols, k) in [(1,8192,1024),(8,8192,1024),(32,8192,1024),(64,8192,1024),
                          (1,1032,64),(32,1032,64),(64,1032,64)]:
        try:
            run_case(B, cols, k, dt)
        except Exception as ex:
            print(f"ERR B={B} cols={cols} k={k} dt={dt}: {repr(ex)[:160]}")

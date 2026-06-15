"""Per-GEMM M=1/M=4 BW efficiency at REAL prod decode shapes (bf16).
This isolates: how close to peak HBM BW does a real decode GEMM get when it
streams its weights at M=1? That efficiency gap (not prefetch) is the lever.
We use the actual DSV3.2 per-layer shapes (TP4 sharded)."""
import torch, sys
torch.manual_seed(0)
dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
PEAK = props.memory_clock_rate*1000.0*2.0*(props.memory_bus_width/8.0)/1e12
print(f"peak={PEAK:.2f} TB/s")

def cbench(fn,nit=80,warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(nit): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/nit

# real per-layer (in,out) shapes for x[M,in] @ W[in,out]; bf16 (2 B/elt)
shapes = [
    ("q_a",   7168, 1536),
    ("q_b",   1536, 6144),
    ("kv",    7168, 576),
    ("o_proj",4096, 7168),
    ("w13",   7168, 4096),
    ("w2",    2048, 7168),
]
for M in [1, 4]:
    print(f"\n--- M={M} per-GEMM bf16 BW (real shapes) ---")
    tot_b=0.0; tot_t=0.0
    for name,K,N in shapes:
        W=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        wb=W.numel()*2
        ms=cbench(lambda: torch.matmul(x,W))
        bw=wb/(ms/1e3)/1e12
        # MoE w13/w2 appear 8x per layer (active experts)
        mult = 8 if name in ("w13","w2") else 1
        tot_b += wb*mult; tot_t += (ms/1e3)*mult
        print(f"  {name:7s} {M}x{K} @ {K}x{N:5d}  {ms*1e3:6.1f} us  read={wb/1e6:5.1f}MB "
              f"BW={bw:4.2f} ({100*bw/PEAK:3.0f}%)  x{mult}")
    layer_bw = tot_b/tot_t/1e12
    print(f"  LAYER total: read={tot_b/1e6:.1f}MB time={tot_t*1e3:.1f}us "
          f"effBW={layer_bw:.2f} TB/s ({100*layer_bw/PEAK:.0f}% of peak)")
print("GEMM_EFF OK")

"""Chain bench v2 — BYTE-FAITHFUL to NVFP4 decode.

Real DSV3.2-REAP decode layer reads ~203 MB of NVFP4 weights (per GPU TP4) and
does trivial M=1 compute. We model the layer as a set of GEMMs whose TOTAL
weight bytes == 203 MB (NVFP4 budget). bf16 reads 2 B/elt, so we size the GEMM
element count = NVFP4_bytes / 2  -> identical 203 MB HBM read, real matmul math.

Schedules over chain of N layers:
  SEQ-fused : run the layer GEMMs back-to-back (PRODUCTION reality: weights
              resident, GEMM streams them). This is the real decode chain.
  SEQ-rw    : explicit "read" (copy 203MB into a live buf) THEN GEMM. Models a
              staged read+compute (pessimistic).
  OVL       : double-buffer — prefetch layer L+1's 203MB on a SIDE stream while
              layer L's GEMMs run on the main stream. ping-pong buffers.
Plus direct CONTENTION test: 1 copy vs 2 concurrent copies on 2 streams ->
does aggregate BW stay at peak (=> they share, no win) or double (=> spare BW)?
"""
import torch, time, sys
torch.manual_seed(0)
dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
PEAK = props.memory_clock_rate * 1000.0 * 2.0 * (props.memory_bus_width / 8.0) / 1e12
print(f"peak_HBM_BW={PEAK:.2f} TB/s SMs={props.multi_processor_count}")

N_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
BYTES_PER_ELT = 0.5

# NVFP4 budget for the real layer
def nvfp4_layer_bytes():
    shapes = [(7168,1536),(1536,6144),(7168,576),(4096,7168)]
    for _ in range(8): shapes += [(7168,4096),(2048,7168)]
    return int(sum(i*o for i,o in shapes) * BYTES_PER_ELT)

NVFP4_BYTES = nvfp4_layer_bytes()
print(f"NVFP4 per-layer read = {NVFP4_BYTES/1e6:.1f} MB ; "
      f"chain = {N_LAYERS*NVFP4_BYTES/1e9:.2f} GB ; M={M}")

# byte-faithful bf16 GEMMs: total bf16 elements = NVFP4_BYTES/2 so HBM read==203MB
# split across a handful of GEMMs with K=7168 (decode hidden) and varying N.
K = 7168
target_elts = NVFP4_BYTES // 2           # bf16 elements to read == NVFP4 bytes
# break into chunks of N so each GEMM is a realistic size
n_gemms = 6
N_each = max(1, target_elts // (K * n_gemms))
gemm_w = [torch.randn(K, N_each, device=dev, dtype=torch.bfloat16) for _ in range(n_gemms)]
xs = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(n_gemms)]
real_compute_bytes = sum(w.numel()*2 for w in gemm_w)
print(f"byte-faithful GEMM: {n_gemms}x [{M}x{K} @ {K}x{N_each}] "
      f"read={real_compute_bytes/1e6:.1f} MB (target {NVFP4_BYTES/1e6:.1f})")

def compute_layer():
    for x,W in zip(xs, gemm_w):
        torch.matmul(x, W)

# raw NVFP4 byte buffers for prefetch + ping-pong live buffers
layer_bytes = [torch.empty(NVFP4_BYTES, dtype=torch.uint8, device=dev) for _ in range(N_LAYERS)]
for b in layer_bytes: b.random_(0,256)
live = [torch.empty(NVFP4_BYTES, dtype=torch.uint8, device=dev) for _ in range(2)]
main = torch.cuda.current_stream(); side = torch.cuda.Stream()

def cbench(fn, nit=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(nit): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/nit

def seq_fused():       # PRODUCTION reality: just the GEMMs, weights resident
    for _ in range(N_LAYERS): compute_layer()

def seq_rw():          # staged read(copy) + compute
    for L in range(N_LAYERS):
        live[L%2].copy_(layer_bytes[L]); compute_layer()

def ovl():             # double-buffer prefetch on side stream
    with torch.cuda.stream(side): live[0].copy_(layer_bytes[0])
    for L in range(N_LAYERS):
        main.wait_stream(side)
        if L+1 < N_LAYERS:
            with torch.cuda.stream(side): live[(L+1)%2].copy_(layer_bytes[L+1])
        compute_layer()

def read_only():
    for L in range(N_LAYERS): live[L%2].copy_(layer_bytes[L])

t_fused = cbench(seq_fused)
t_rw    = cbench(seq_rw)
t_ovl   = cbench(ovl)
t_comp  = cbench(seq_fused)   # same as fused (compute-only IS the prod chain)
t_read  = cbench(read_only)

chain_b = N_LAYERS*NVFP4_BYTES
def bw1(ms): return chain_b/(ms/1e3)/1e12          # 1x read (GEMM streams once)
def bw2(ms): return 2*chain_b/(ms/1e3)/1e12        # copy = 2x

print(f"\n=== chain {N_LAYERS} layers, M={M} (byte-faithful) ===")
print(f"SEQ-fused (prod: GEMM only) : {t_fused:.3f} ms ({t_fused/N_LAYERS*1e3:.1f} us/L) "
      f"BW={bw1(t_fused):.2f} TB/s ({100*bw1(t_fused)/PEAK:.0f}%)")
print(f"SEQ-rw  (copy-read+compute) : {t_rw:.3f} ms ({t_rw/N_LAYERS*1e3:.1f} us/L)")
print(f"OVL  (prefetch L+1 || cmp L): {t_ovl:.3f} ms ({t_ovl/N_LAYERS*1e3:.1f} us/L)")
print(f"read-only (copy 203MB/L)    : {t_read:.3f} ms ({t_read/N_LAYERS*1e3:.1f} us/L) "
      f"BW(2x)={bw2(t_read):.2f} TB/s ({100*bw2(t_read)/PEAK:.0f}%)")
print(f"\nOVL vs SEQ-rw : {t_ovl:.3f} vs {t_rw:.3f} -> {t_rw/t_ovl:.3f}x ({100*(1-t_ovl/t_rw):.1f}%)")
print(f"OVL vs SEQ-fused(prod): {t_ovl:.3f} vs {t_fused:.3f} -> {t_fused/t_ovl:.3f}x")
print(f"ideal SEQ-rw=read+compute={t_read+t_comp:.3f} ; ideal OVL=max(read,compute)={max(t_read,t_comp):.3f}")

# ---- DIRECT HBM CONTENTION TEST ----
print("\n=== contention: 1 vs 2 concurrent copies on 2 streams ===")
s2 = torch.cuda.Stream()
big = torch.empty(NVFP4_BYTES, dtype=torch.uint8, device=dev); big.random_(0,256)
dst1 = torch.empty_like(big); dst2 = torch.empty_like(big)
def one_copy():
    dst1.copy_(big)
def two_copy():
    with torch.cuda.stream(side): dst1.copy_(big)
    with torch.cuda.stream(s2):   dst2.copy_(big)
    main.wait_stream(side); main.wait_stream(s2)
t1 = cbench(one_copy, nit=50); t2 = cbench(two_copy, nit=50)
bw_1 = 2*NVFP4_BYTES/(t1/1e3)/1e12
bw_2 = 2*2*NVFP4_BYTES/(t2/1e3)/1e12   # two copies = 2x the bytes
print(f"1 copy : {t1*1e3:.1f} us  BW(2x)={bw_1:.2f} TB/s ({100*bw_1/PEAK:.0f}%)")
print(f"2 copy : {t2*1e3:.1f} us  aggregate_BW={bw_2:.2f} TB/s ({100*bw_2/PEAK:.0f}%)")
print(f"  -> if 2-copy BW ~= 1-copy BW, HBM is SATURATED (no spare BW for prefetch)")
print(f"  -> 2-copy took {t2/t1:.2f}x the time of 1-copy (1.0=free overlap, 2.0=fully serialized)")
print("CHAIN2 OK")

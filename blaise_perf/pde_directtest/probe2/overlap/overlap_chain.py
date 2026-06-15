"""Full chain bench: sequential vs double-buffered (prefetch) decode.

Setup: a chain of N_LAYERS stand-in decode layers at prod weight sizes
(per GPU under TP4, NVFP4 ~0.5 B/elt). Per layer:
  attn: q_a 7168x1536, q_b 1536x6144, kv 7168x576, o_proj 4096x7168
  moe : 8 active experts x (w13 7168x4096 + w2 2048x7168)

Two views of the layer weights:
 (A) REAL GEMMs in bf16 at prod shapes -> the "compute" is the real decode
     matmul; at M=1 it is BW-bound and IS the weight read.
 (B) RAW NVFP4 byte buffers (uint8, exact 0.5 B/elt size) streamed by a
     copy -> models the true NVFP4 weight-read time (the thing a prefetch
     would move).

Schedules over the chain:
  SEQ : for layer L: read weights of L (copy bytes to live buffer) ; compute L.
  OVL : double-buffer: while computing layer L on main stream, prefetch
        layer L+1's bytes on a SIDE stream into the other ping-pong buffer.
  We also test SEQ-fused (just run the real GEMMs back-to-back, no explicit
  copy) as the production reality (weights already resident), and the
  pure-compute chain time.

Crux measured: does prefetch of L+1 overlap with USEFUL work of L, or do both
contend for HBM BW (=> ~0 gain)? We report achieved BW vs 7.67 TB/s peak.
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
print(f"N_LAYERS={N_LAYERS} M={M} dtype-for-read=NVFP4(0.5B/elt)")

# ---- prod shapes (out_dim, in_dim) so x[M,in] @ W[in,out] ----
def layer_gemm_shapes():
    # (in, out) for x@W
    shapes = [
        (7168, 1536),        # q_a
        (1536, 24576 // 4),  # q_b  (6144)
        (7168, 576),         # kv
        (16384 // 4, 7168),  # o_proj (4096 in, 7168 out)
    ]
    for _ in range(8):       # 8 active experts
        shapes.append((7168, 4096))   # w13 (gate+up fused dim already in 4096? keep as-is)
        shapes.append((2048, 7168))   # w2
    return shapes

shapes = layer_gemm_shapes()
tot_elts = sum(i * o for i, o in shapes)
weight_bytes = int(tot_elts * BYTES_PER_ELT)
print(f"per-layer elts={tot_elts/1e9:.3f}B  NVFP4_bytes={weight_bytes/1e6:.1f} MB  "
      f"chain_bytes={N_LAYERS*weight_bytes/1e9:.2f} GB")

# ---- (A) real bf16 GEMM weights per layer (resident) ----
# To keep memory bounded we share ONE set of weight tensors per distinct shape
# across layers for the compute, but for the prefetch we need per-layer byte
# buffers. Memory: chain raw bytes = N*203MB ~ 3.2GB @16 layers; fine on 183GB.
gemm_w = [torch.randn(i, o, device=dev, dtype=torch.bfloat16) for i, o in shapes]
xs = [torch.randn(M, i, device=dev, dtype=torch.bfloat16) for i, o in shapes]

def compute_layer():
    # run the real decode GEMMs of one layer (weights resident)
    for x, W in zip(xs, gemm_w):
        torch.matmul(x, W)

# ---- (B) raw NVFP4 byte buffers per layer + ping-pong live buffers ----
layer_bytes = [torch.empty(weight_bytes, dtype=torch.uint8, device=dev)
               for _ in range(N_LAYERS)]
for b in layer_bytes:
    b.random_(0, 256)
# ping-pong "live" weight buffers the compute would consume after a read
live = [torch.empty(weight_bytes, dtype=torch.uint8, device=dev) for _ in range(2)]

main = torch.cuda.current_stream()
side = torch.cuda.Stream()


def cuda_bench(fn, nit=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(nit):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / nit  # ms


# ---------- schedule 1: SEQ (read bytes then compute, per layer) ----------
def run_seq():
    for L in range(N_LAYERS):
        live[L % 2].copy_(layer_bytes[L])   # READ weights (HBM read+write)
        compute_layer()                      # COMPUTE (BW-bound real GEMMs)

# ---------- schedule 2: OVL (double-buffered prefetch on side stream) -----
def run_ovl():
    # prefetch layer 0 first
    with torch.cuda.stream(side):
        live[0].copy_(layer_bytes[0])
    for L in range(N_LAYERS):
        cur = L % 2
        # make main wait until this layer's weights are staged
        main.wait_stream(side)
        # kick prefetch of L+1 on side stream NOW (concurrent with compute)
        if L + 1 < N_LAYERS:
            with torch.cuda.stream(side):
                live[(L + 1) % 2].copy_(layer_bytes[L + 1])
        # compute layer L on main stream (overlaps with side prefetch)
        compute_layer()

# ---------- schedule 3: compute-only (weights resident, no read) ----------
def run_compute_only():
    for L in range(N_LAYERS):
        compute_layer()

# ---------- schedule 4: read-only chain (no compute) ----------
def run_read_only():
    for L in range(N_LAYERS):
        live[L % 2].copy_(layer_bytes[L])


t_seq = cuda_bench(run_seq)
t_ovl = cuda_bench(run_ovl)
t_comp = cuda_bench(run_compute_only)
t_read = cuda_bench(run_read_only)

chain_read_bytes = N_LAYERS * weight_bytes
# copy moves 2x (read+write); the *read* component is 1x. report read-equivalent.
def bw_read(ms):  # treat copy as 2x traffic
    return 2 * chain_read_bytes / (ms / 1e3) / 1e12

print(f"\n=== chain over {N_LAYERS} layers, M={M} ===")
print(f"SEQ  (read+compute) : {t_seq:.3f} ms  ({t_seq/N_LAYERS*1e3:.1f} us/layer)")
print(f"OVL  (prefetch||cmp): {t_ovl:.3f} ms  ({t_ovl/N_LAYERS*1e3:.1f} us/layer)")
print(f"  -> OVL speedup vs SEQ: {t_seq/t_ovl:.3f}x  (delta {100*(1-t_ovl/t_seq):.1f}%)")
print(f"compute-only        : {t_comp:.3f} ms  ({t_comp/N_LAYERS*1e3:.1f} us/layer)")
print(f"read-only (copy)    : {t_read:.3f} ms  ({t_read/N_LAYERS*1e3:.1f} us/layer)  "
      f"BW(2x)={bw_read(t_read):.2f} TB/s ({100*bw_read(t_read)/PEAK:.0f}%)")
print(f"\nidealized: max(read,compute)={max(t_read,t_comp):.3f} ms  "
      f"sum(read+compute)={t_read+t_comp:.3f} ms")
print(f"SEQ vs sum(read+compute): {t_seq:.3f} vs {t_read+t_comp:.3f}")
print(f"OVL vs max(read,compute): {t_ovl:.3f} vs {max(t_read,t_comp):.3f}  "
      f"(if OVL≈max, overlap worked; if OVL≈sum, it didn't)")
print("CHAIN OK")

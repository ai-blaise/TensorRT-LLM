"""Minimal 2-layer decode weight-prefetch overlap probe (validation run).

Question: at decode M=1, per-layer latency is weight-HBM-read bound.
Can double-buffering (prefetch layer N+1 weights on a side stream while
layer N computes) hide the weight-read latency, or does HBM BW saturation
make overlap futile?

This MINIMAL version: 2 layers, validate the mechanism + measure achieved BW.
"""
import torch, time, math

torch.manual_seed(0)
dev = torch.device("cuda:0")
assert torch.cuda.is_available()
props = torch.cuda.get_device_properties(0)
# peak HBM BW from clock * bus: memory_clock_rate is in kHz (per pin, DDR -> x2)
mem_clk_hz = props.memory_clock_rate * 1000.0  # kHz -> Hz
bus_bits = props.memory_bus_width
peak_bw = mem_clk_hz * 2.0 * (bus_bits / 8.0)  # GDDR/HBM DDR factor 2
print(f"GPU {props.name} SMs={props.multi_processor_count} "
      f"peak_HBM_BW={peak_bw/1e12:.2f} TB/s")

# ---- prod weight sizing per layer per GPU under TP4, NVFP4 (~0.5 B/elt) ----
BYTES_PER_ELT = 0.5  # NVFP4 packed (4-bit) ; we allocate uint8 -> 2 elts/byte


def elts_per_layer():
    # attention proj (TP4 shards on the appropriate dim)
    q_a = 7168 * 1536
    q_b = 1536 * (24576 // 4)
    kv = 7168 * 576
    o_proj = (16384 // 4) * 7168
    attn = q_a + q_b + kv + o_proj
    # MoE active experts: decode B=1 top-8 of 128 ; per expert w13 + w2
    w13 = 7168 * 4096
    w2 = 2048 * 7168
    per_expert = w13 + w2
    moe_active = 8 * per_expert
    return attn + moe_active, attn, moe_active


tot_elts, attn_elts, moe_elts = elts_per_layer()
weight_bytes = int(tot_elts * BYTES_PER_ELT)
print(f"per-layer active elts={tot_elts/1e9:.3f}B  "
      f"attn={attn_elts/1e6:.0f}M moe={moe_elts/1e9:.3f}B  "
      f"weight_bytes={weight_bytes/1e6:.1f} MB  "
      f"(attn {attn_elts*BYTES_PER_ELT/1e6:.1f} + moe {moe_elts*BYTES_PER_ELT/1e6:.1f})")

N_LAYERS = 2
# allocate per-layer raw weight byte buffers (uint8) of exact size
wbufs = [torch.empty(weight_bytes, dtype=torch.uint8, device=dev) for _ in range(N_LAYERS)]
for w in wbufs:
    w.random_(0, 256)

# two ping-pong staging buffers for overlapped schedule (same size)
stage = [torch.empty(weight_bytes, dtype=torch.uint8, device=dev) for _ in range(2)]

# a tiny activation for a representative GEMM at M=1
M = 1
HID = 7168
x = torch.randn(M, HID, device=dev, dtype=torch.bfloat16)
# a compute weight that actually multiplies (small, bf16) -- the "useful work"
# represent attention o_proj-ish gemm 7168x7168 in bf16 just as a stand-in compute op
gemm_w = torch.randn(HID, HID, device=dev, dtype=torch.bfloat16)

# "read weights" kernel: stream the weight bytes. Use a reduction (sum) which
# forces an actual HBM read of every byte. .sum() over uint8 -> int64 accumulate.
def read_weights(buf):
    return buf.sum()

def compute(x):
    return torch.matmul(x, gemm_w)

# ---- validate the read kernel actually reads at HBM speed (BW sanity) ----
torch.cuda.synchronize()
NIT = 50
# warm
for _ in range(5):
    _ = read_weights(wbufs[0])
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(NIT):
    s = read_weights(wbufs[0])
torch.cuda.synchronize()
t1 = time.perf_counter()
read_ms = (t1 - t0) / NIT * 1e3
read_bw = weight_bytes / ((t1 - t0) / NIT) / 1e12
print(f"[read-only sum] {read_ms:.3f} ms/layer  achieved_BW={read_bw:.2f} TB/s "
      f"({100*read_bw/(peak_bw/1e12):.0f}% of peak)")

# also a pure copy (D2D) as alternative read proxy
for _ in range(5):
    stage[0].copy_(wbufs[0])
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(NIT):
    stage[0].copy_(wbufs[0])
torch.cuda.synchronize()
t1 = time.perf_counter()
copy_ms = (t1 - t0) / NIT * 1e3
# copy reads weight_bytes + writes weight_bytes = 2x traffic
copy_bw = 2 * weight_bytes / ((t1 - t0) / NIT) / 1e12
print(f"[copy D2D] {copy_ms:.3f} ms/layer  achieved_BW(2x)={copy_bw:.2f} TB/s "
      f"({100*copy_bw/(peak_bw/1e12):.0f}% of peak)")

# ---- compute-only timing at M=1 ----
for _ in range(5):
    _ = compute(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(NIT):
    y = compute(x)
torch.cuda.synchronize()
t1 = time.perf_counter()
comp_ms = (t1 - t0) / NIT * 1e3
print(f"[compute M=1 7168x7168] {comp_ms:.4f} ms  "
      f"(ratio compute/read = {comp_ms/read_ms:.3f})")

print("MIN VALIDATION OK")

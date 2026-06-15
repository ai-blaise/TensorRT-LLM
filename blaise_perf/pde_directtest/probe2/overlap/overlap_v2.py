"""V2: faithful weight-read kernels + real M=1 GEMMs.

Finding from v1: uint8.sum() is conversion-bound (5% BW), NOT a valid weight
read. D2D copy hits 81% peak = the real BW-bound proxy. But the MOST faithful
decode primitive is a real M=1 GEMM that READS the full weight matrix and does
the (tiny) decode compute. At M=1, FLOPs = 2*K*N is trivial; bytes read = K*N*elt.
So a real GEMM at M=1 IS the weight read. We measure:
  (a) achieved BW of a real bf16 M=1 GEMM (how close to peak does the HW get
      when "reading weights" via a GEMM?)
  (b) whether there is ANY compute to hide a prefetch behind.
"""
import torch, time
torch.manual_seed(0)
dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
peak_bw = props.memory_clock_rate * 1000.0 * 2.0 * (props.memory_bus_width / 8.0)
print(f"peak_HBM_BW={peak_bw/1e12:.2f} TB/s SMs={props.multi_processor_count}")


def bench(fn, nit=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(nit):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / nit


def cuda_event_bench(fn, nit=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(nit)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(nit)]
    for i in range(nit):
        ev0[i].record()
        fn()
        ev1[i].record()
    torch.cuda.synchronize()
    return sum(a.elapsed_time(b) for a, b in zip(ev0, ev1)) / nit  # ms


# A real bf16 weight matrix sized so its read ~= a chunk of the layer budget.
# Use a single big GEMM weight to probe peak M=1 GEMM BW.
HID = 7168
for N in [7168, 16384, 28672]:
    W = torch.randn(HID, N, device=dev, dtype=torch.bfloat16)
    x = torch.randn(1, HID, device=dev, dtype=torch.bfloat16)
    wbytes = W.numel() * 2
    ms = cuda_event_bench(lambda: torch.matmul(x, W))
    bw = wbytes / (ms / 1e3) / 1e12
    print(f"[M=1 bf16 GEMM 1x{HID} @ {HID}x{N}] {ms*1e3:.1f} us  "
          f"read={wbytes/1e6:.0f}MB  BW={bw:.2f} TB/s ({100*bw/(peak_bw/1e12):.0f}%)")

print("---- M=4 ----")
for N in [7168, 28672]:
    W = torch.randn(HID, N, device=dev, dtype=torch.bfloat16)
    x = torch.randn(4, HID, device=dev, dtype=torch.bfloat16)
    wbytes = W.numel() * 2
    ms = cuda_event_bench(lambda: torch.matmul(x, W))
    bw = wbytes / (ms / 1e3) / 1e12
    print(f"[M=4 bf16 GEMM 4x{HID} @ {HID}x{N}] {ms*1e3:.1f} us  BW={bw:.2f} TB/s "
          f"({100*bw/(peak_bw/1e12):.0f}%)")

# fp8 GEMM (closer to NVFP4 byte density: 1 B/elt vs 0.5; still tests M=1 BW)
print("---- M=1 fp8 (1 B/elt) ----")
try:
    for N in [7168, 28672]:
        Wf = torch.randn(HID, N, device=dev).to(torch.float8_e4m3fn)
        xf = torch.randn(1, HID, device=dev).to(torch.float8_e4m3fn)
        wbytes = Wf.numel() * 1
        # torch._scaled_mm path
        scale = torch.tensor(1.0, device=dev)
        def f():
            return torch._scaled_mm(xf, Wf.t(), scale_a=scale, scale_b=scale,
                                    out_dtype=torch.bfloat16)
        try:
            f()
        except Exception as e:
            print("  scaled_mm err", repr(e)[:120]); break
        ms = cuda_event_bench(f)
        bw = wbytes / (ms / 1e3) / 1e12
        print(f"[M=1 fp8 1x{HID} @ {HID}x{N}] {ms*1e3:.1f} us read={wbytes/1e6:.0f}MB "
              f"BW={bw:.2f} TB/s ({100*bw/(peak_bw/1e12):.0f}%)")
except Exception as e:
    print("fp8 path failed", repr(e)[:160])

print("V2 OK")

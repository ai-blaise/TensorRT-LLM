import torch, subprocess, sys
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("dev", torch.cuda.get_device_name(0))
p = torch.cuda.get_device_properties(0)
print("SMs", p.multi_processor_count, "mem_GB", round(p.total_memory/1e9,1))
# memory clock / bus width if available
for a in ["memory_clock_rate","memory_bus_width","l2_cache_size"]:
    print(a, getattr(p, a, "n/a"))
try:
    print("nvcc:", subprocess.check_output(["nvcc","--version"]).decode().strip().splitlines()[-1])
except Exception as e:
    print("nvcc err", e)
# fp8/fp4 dtypes present?
print("has float8_e4m3fn", hasattr(torch, "float8_e4m3fn"))
print("has uint8", hasattr(torch, "uint8"))

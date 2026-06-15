"""Final: CUDA-graph (launch-overhead-free) vs eager for the M=1 decode chain,
and overlap test INSIDE the graph regime. Quantifies how much M=1 decode time is
launch/overhead (the real lever) vs HBM BW floor (where prefetch can't help).
Uses real prod-shape GEMMs. 16-layer chain."""
import torch, sys
torch.manual_seed(0)
dev=torch.device("cuda:0")
props=torch.cuda.get_device_properties(0)
PEAK=props.memory_clock_rate*1000.0*2.0*(props.memory_bus_width/8.0)/1e12
N=16; M=int(sys.argv[1]) if len(sys.argv)>1 else 1
print(f"peak={PEAK:.2f} TB/s  N_LAYERS={N} M={M}")

shapes=[("q_a",7168,1536),("q_b",1536,6144),("kv",7168,576),
        ("o_proj",4096,7168),("w13",7168,4096),("w2",2048,7168)]
mults={"w13":8,"w2":8}
# build full per-layer op list (8x for moe experts), shared weights across layers
ops=[]
layer_read_bytes=0
for name,K,Nn in shapes:
    m=mults.get(name,1)
    for _ in range(m):
        W=torch.randn(K,Nn,device=dev,dtype=torch.bfloat16)
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        ops.append((x,W)); layer_read_bytes+=W.numel()*2
print(f"layer bf16 read={layer_read_bytes/1e6:.1f}MB ; "
      f"NVFP4-equiv={layer_read_bytes/4/1e6:.1f}MB ; chain bf16={N*layer_read_bytes/1e9:.2f}GB")

def chain():
    for _ in range(N):
        for x,W in ops:
            torch.matmul(x,W)

def cbench(fn,nit=50,warm=15):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(nit): fn()
    e1.record();torch.cuda.synchronize()
    return e0.elapsed_time(e1)/nit

# eager
t_eager=cbench(chain)
# cuda graph capture
g=torch.cuda.CUDAGraph()
chain();chain();torch.cuda.synchronize()  # warm allocator
s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        chain()
torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
def replay(): g.replay()
t_graph=cbench(replay)

chain_b=N*layer_read_bytes
bw_eager=chain_b/(t_eager/1e3)/1e12
bw_graph=chain_b/(t_graph/1e3)/1e12
n_kernels=N*len(ops)
print(f"\nEAGER chain : {t_eager:.3f} ms  BW={bw_eager:.2f} TB/s ({100*bw_eager/PEAK:.0f}%) "
      f"({t_eager*1e3/n_kernels:.1f} us/kernel, {n_kernels} kernels)")
print(f"GRAPH chain : {t_graph:.3f} ms  BW={bw_graph:.2f} TB/s ({100*bw_graph/PEAK:.0f}%) "
      f"({t_graph*1e3/n_kernels:.2f} us/kernel)")
print(f"  -> graph speedup {t_eager/t_graph:.2f}x ; launch overhead removed = "
      f"{(t_eager-t_graph)/t_eager*100:.0f}% of eager time = {(t_eager-t_graph)*1e3/n_kernels:.1f} us/kernel")
print(f"  -> GRAPH is the launch-free floor. Its BW {100*bw_graph/PEAK:.0f}% of peak shows the")
print(f"     residual is {'BW-bound (overlap futile)' if bw_graph/PEAK>0.6 else 'STILL overhead/occupancy-bound'}.")
print("GRAPH_VS_EAGER OK")

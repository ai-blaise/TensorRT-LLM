"""Overhead-bound probe: isolate the launch-overhead win the piecewise capture
delivers, by shrinking per-op compute so latency == kernel-launch overhead.

The 94%-fewer-launches structural win (gate_definitive Tier 1) converts to a
large latency win in the launch-bound regime -- the real DSA prefill is
overhead-bound (project memory: prefill/decode runs 20-40x the bw floor,
launch/dispatch-overhead dominated), unlike the dense-GEMM Tier-2 stand-in which
is compute-bound and so understates the win. Same op STRUCTURE as the real DSA
decoder span; only per-op tensor width is shrunk to expose launch overhead.
"""
import torch
import torch.nn as nn
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.utils import set_capture_piecewise_cuda_graph_flag
from tensorrt_llm._torch.utils import (model_extra_attrs,
    set_per_request_piecewise_cuda_graph_flag, set_piecewise_cuda_graph_flag)

DEV="cuda"; DT=torch.bfloat16; H=512; NT=64; NLAYERS=16

class M(nn.Module):
    def __init__(self,n):
        super().__init__(); self.n=n
        self.w=[[torch.randn(H,H,device=DEV,dtype=DT)*0.02 for _ in range(6)] for _ in range(n)]
    def forward(self,x):
        h=x
        for i in range(self.n):
            r=h
            for w in self.w[i]: h=torch.relu(h@w)
            h=h+r
        return h

def time_ms(fn,args,it=100,wu=20):
    for _ in range(wu): fn(*args)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn(*args)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it

m=M(NLAYERS).eval()
xd=torch.randn(NT,H,device=DEV,dtype=DT); x=xd.clone(); attrs={}
with torch.no_grad(): ref=m(x).clone()
eager=time_ms(lambda a: m(a),(x,))
print(f"per-op-tiny: H={H} NT={NT} layers={NLAYERS}  EAGER={eager:.4f} ms")

set_piecewise_cuda_graph_flag(True); set_per_request_piecewise_cuda_graph_flag(True); set_capture_piecewise_cuda_graph_flag(True)
be=Backend(enable_inductor=False,enable_userbuffers=False,enable_piecewise_cuda_graph=True,capture_num_tokens=[NT],max_num_streams=1,mapping=None)
class W(nn.Module):
    def __init__(s,i): super().__init__(); s.inner=i
    def forward(s,input_ids): return s.inner(input_ids)
c=torch.compile(W(m).eval(),backend=be,fullgraph=False)
def pw_call(a):
    with model_extra_attrs(attrs), torch.no_grad(): return c(a)
with model_extra_attrs(attrs), torch.no_grad():
    for _ in range(6):
        x.copy_(xd); o=c(x)
    torch.cuda.synchronize(); x.copy_(xd); o=c(x).clone(); torch.cuda.synchronize()
exact=torch.equal(ref,o)
pw=time_ms(pw_call,(x,))
print(f"per-op-tiny:                       PIECEWISE={pw:.4f} ms  bit-exact={exact}")
print(f">> overhead-bound speedup eager->piecewise: {eager/pw:.2f}x "
      f"(launch-overhead win the 94%-fewer-launches result converts to when "
      f"ops are launch-bound, as in real prefill)")
print("OVERHEAD_PROBE_DONE")

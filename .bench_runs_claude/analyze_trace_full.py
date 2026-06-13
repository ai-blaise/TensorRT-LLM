#!/usr/bin/env python3
"""In-depth kineto-trace breakdown: where every microsecond goes, per decode iter.

Streams the chrome-trace 3-line event blocks. Buckets events by phase category
(GPU kernel / memcpy / memset / host cuda_runtime / cpu_op), accumulates per-name
count+dur, per-GPU-stream busy time, and the step span, then prints:
  - top GPU kernels by total time (per-iter us)
  - GPU time grouped into subsystems (regex on demangled name)
  - top host cuda_runtime ops (the sync/launch gaps)
  - stream overlap + idle estimate
iters inferred from cudaGraphLaunch count (1/iter)."""
import re, sys
from collections import defaultdict

NAME_RE = re.compile(r'"name":\s*"((?:[^"\\]|\\.)*)"')
DUR_RE  = re.compile(r'"dur":\s*([0-9.]+)')
TS_RE   = re.compile(r'"ts":\s*([0-9.]+)')
CAT_RE  = re.compile(r'"cat":\s*"([^"]*)"')
TID_RE  = re.compile(r'"tid":\s*(-?[0-9]+)')

# subsystem buckets (first match wins), by substring/regex on the demangled kernel name
SUBSYS = [
 ("MoE a2a/comm",   re.compile(r'moe.*[Cc]omm|[Aa]ll.?[Tt]o.?[Aa]ll|alltoall|fused_moe.*comm|moe_prepare|CounterComm|Fifo|nvshmem|mnnvl', re.I)),
 ("MoE expert GEMM",re.compile(r'moe.*gemm|grouped.*gemm|group.*gemm|expert|warp.?decode|MoeGemm|cutlass.*group', re.I)),
 ("Dense/proj GEMM", re.compile(r'nvjet|cutlass|gemm|Kernel.*Gemm|sm100|sm90.*gemm|fp4.*gemm|cublas|bmm|matmul', re.I)),
 ("Attention/MLA",  re.compile(r'attention|mla|flash|mha|mqa|paged|fmha|decoderMaskedMulti|attn', re.I)),
 ("Indexer/HISA",   re.compile(r'indexer|hisa|topk|top_k|paged_mqa_logits|fp8_index|sparse', re.I)),
 ("Quant",          re.compile(r'quantize|quant_|nvfp4_quant|fp4_quant|block_size|cvt_.*fp4|dequant|scaled', re.I)),
 ("Norm",           re.compile(r'rms_?norm|layer_?norm|norm_|gated_norm|add_rmsnorm', re.I)),
 ("KVarN/cache",    re.compile(r'kvarn|kv_cache|cache.*restore|paged.*kv|reshape.*cache|copy.*kv', re.I)),
 ("Elementwise/copy",re.compile(r'elementwise|copy|memcpy|memset|cat_|index_|gather|scatter|fill|cast|add_|mul_|silu|gelu|activation|rope|RoPE|embedding', re.I)),
 ("Reduce/comm-coll",re.compile(r'reduce|allreduce|nccl|all_reduce|ncclDevKernel', re.I)),
]
def classify(name):
    for label, rx in SUBSYS:
        if rx.search(name): return label
    return "Other"

def analyze(path):
    kern = defaultdict(lambda:[0,0.0])      # name -> [count, dur]
    host = defaultdict(lambda:[0,0.0])      # cuda_runtime name -> [count, dur]
    cpuop = defaultdict(lambda:[0,0.0])     # cpu_op name -> [count, dur]
    stream_busy = defaultdict(float)        # tid -> dur (GPU streams only)
    sub = defaultdict(lambda:[0,0.0])       # subsystem -> [count, dur]
    gpu_min_ts=[None]; gpu_max_end=[0.0]
    iters = 0
    pend_cat=None; pend_name=None; pend_ts=None
    GPU_CATS={"kernel","gpu_memcpy","gpu_memset"}
    with open(path, errors="replace") as f:
        for line in f:
            if '"cat":' in line and '"name":' in line:
                c=CAT_RE.search(line); n=NAME_RE.search(line)
                pend_cat = c.group(1) if c else None
                pend_name= n.group(1) if n else None
                t=TS_RE.search(line); pend_ts=float(t.group(1)) if t else None
                # ts/dur sometimes on same line
                d=DUR_RE.search(line)
                if d and pend_cat: _record(pend_cat,pend_name,float(d.group(1)),pend_ts,line,kern,host,cpuop,stream_busy,sub,gpu_min_ts,gpu_max_end,GPU_CATS); pend_cat=None
                continue
            if pend_cat is not None:
                d=DUR_RE.search(line); t=TS_RE.search(line)
                ts = float(t.group(1)) if t else pend_ts
                if d: _record(pend_cat,pend_name,float(d.group(1)),ts,line,kern,host,cpuop,stream_busy,sub,gpu_min_ts,gpu_max_end,GPU_CATS)
                pend_cat=None
    iters = host.get("cudaGraphLaunch",[0])[0] or 1
    return kern,host,cpuop,stream_busy,sub,iters,gpu_min_ts[0],gpu_max_end[0]

def _record(cat,name,dur,ts,line,kern,host,cpuop,stream_busy,sub,gmin,gmax,GPU_CATS):
    if name=="cudaGraphLaunch": host[name][0]+=1; host[name][1]+=dur
    if cat in GPU_CATS:
        kern[name][0]+=1; kern[name][1]+=dur
        s=classify(name); sub[s][0]+=1; sub[s][1]+=dur
        ti=TID_RE.search(line)
        if ti: stream_busy[ti.group(1)]+=dur
        if ts is not None:
            if gmin[0] is None or ts<gmin[0]: gmin[0]=ts
            if ts+dur>gmax[0]: gmax[0]=ts+dur
    elif cat=="cuda_runtime":
        if name!="cudaGraphLaunch": host[name][0]+=1; host[name][1]+=dur
    elif cat=="cpu_op":
        cpuop[name][0]+=1; cpuop[name][1]+=dur

def main():
    path=sys.argv[1]
    kern,host,cpuop,stream_busy,sub,iters,gmin,gmax = analyze(path)
    span_ms = (gmax-gmin)/1000.0 if gmin is not None else 0
    tot_gpu = sum(v[1] for v in kern.values())
    print(f"trace={path}\niters(window)={iters}  span={span_ms:.1f}ms  span/iter={span_ms*1000/iters:.0f}us")
    print(f"total GPU kernel-time(sum dur)={tot_gpu/1000:.1f}ms  per-iter={tot_gpu/iters:.0f}us  overlap=sum/span={tot_gpu/1000/span_ms:.2f}x" if span_ms else "")
    busiest = max(stream_busy.values()) if stream_busy else 0
    print(f"busiest single GPU stream busy/iter={busiest/iters:.0f}us  -> approx GPU-bound floor; idle vs span ~ {span_ms*1000/iters - busiest/iters:.0f}us/iter")
    print(f"\n=== GPU time by SUBSYSTEM (per-iter us, sorted) ===")
    print(f"{'subsystem':<20}{'us/iter':>10}{'%GPU':>8}{'launches/iter':>15}")
    for s,(c,d) in sorted(sub.items(), key=lambda x:-x[1][1]):
        print(f"{s:<20}{d/iters:>10.0f}{100*d/tot_gpu:>7.1f}%{c/iters:>15.0f}")
    print(f"\n=== TOP 30 GPU kernels by total time (per-iter us) ===")
    print(f"{'us/iter':>9}{'launch/it':>10}{'us/call':>9}  kernel")
    for name,(c,d) in sorted(kern.items(), key=lambda x:-x[1][1])[:30]:
        print(f"{d/iters:>9.0f}{c/iters:>10.1f}{d/c:>9.2f}  {name[:88]}")
    print(f"\n=== TOP 15 HOST cuda_runtime ops by total time (per-iter us) ===")
    print(f"{'us/iter':>9}{'calls/it':>10}{'us/call':>9}  op")
    for name,(c,d) in sorted(host.items(), key=lambda x:-x[1][1])[:15]:
        print(f"{d/iters:>9.0f}{c/iters:>10.1f}{d/c:>9.2f}  {name[:60]}")
    print(f"\n=== TOP 12 CPU aten ops by total time (per-iter us) — host 'glue' ===")
    for name,(c,d) in sorted(cpuop.items(), key=lambda x:-x[1][1])[:12]:
        print(f"{d/iters:>9.0f}{c/iters:>10.1f}  {name[:70]}")

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Rigorous GPU busy/idle/bubble analysis of a kineto trace.

Answers: is the GPU actually ~100% busy, or are there recoverable bubbles (e.g.
the GPU idling while the host blocks in cudaEventSynchronize / launches a graph)?

Collects every GPU-stream interval (cat kernel/gpu_memcpy/gpu_memset, keyed by
tid=stream), merges them into a busy-UNION across all streams, and compares to
the step span -> true GPU-busy %. Then finds the largest idle gaps and checks
whether host cuda_runtime ops (cudaEventSynchronize, cudaGraphLaunch) overlap
them. 3-line chrome-trace event blocks: name line carries cat/name/tid; the
NEXT line carries ts/dur."""
import re, sys
from collections import defaultdict

NAME_RE = re.compile(r'"name":\s*"((?:[^"\\]|\\.)*)"')
CAT_RE  = re.compile(r'"cat":\s*"([^"]*)"')
TID_RE  = re.compile(r'"tid":\s*(-?[0-9]+)')
TS_RE   = re.compile(r'"ts":\s*([0-9.]+)')
DUR_RE  = re.compile(r'"dur":\s*([0-9.]+)')
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

def analyze(path):
    gpu = []                       # (ts, ts+dur) for all GPU streams
    per_stream = defaultdict(lambda:[0,0.0])  # tid -> [count, dur]
    host = defaultdict(list)       # name -> [(ts, ts+dur)] for cudaEventSync/cudaGraphLaunch
    iters = 0
    pend = None
    with open(path, errors="replace") as f:
        for line in f:
            if '"cat":' in line and '"name":' in line:
                c = CAT_RE.search(line); n = NAME_RE.search(line); ti = TID_RE.search(line)
                cat = c.group(1) if c else ""
                name = n.group(1) if n else ""
                tid = ti.group(1) if ti else "?"
                pend = (cat, name, tid)
                # ts/dur may already be on this line
                ts = TS_RE.search(line); du = DUR_RE.search(line)
                if ts and du:
                    _rec(cat, name, tid, float(ts.group(1)), float(du.group(1)), gpu, per_stream, host)
                    pend = None
                continue
            if pend is not None:
                ts = TS_RE.search(line); du = DUR_RE.search(line)
                if ts and du:
                    _rec(*pend, float(ts.group(1)), float(du.group(1)), gpu, per_stream, host)
                pend = None
    iters = len(host.get("cudaGraphLaunch", [])) or 1
    return gpu, per_stream, host, iters

def _rec(cat, name, tid, ts, dur, gpu, per_stream, host):
    if cat in GPU_CATS:
        gpu.append((ts, ts+dur)); per_stream[tid][0]+=1; per_stream[tid][1]+=dur
    elif cat == "cuda_runtime" and name in ("cudaEventSynchronize","cudaGraphLaunch","cudaStreamSynchronize"):
        host[name].append((ts, ts+dur))

def merge(intervals):
    if not intervals: return [], 0.0
    s = sorted(intervals); out=[list(s[0])];
    for a,b in s[1:]:
        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a,b])
    busy = sum(b-a for a,b in out)
    return out, busy

def main():
    path = sys.argv[1]
    gpu, per_stream, host, iters = analyze(path)
    merged, busy = merge(gpu)
    span = merged[-1][1] - merged[0][0] if merged else 0
    idle = span - busy
    print(f"trace={path}  iters={iters}")
    print(f"span={span/1000:.1f}ms ({span/iters:.0f}us/iter)  GPU-busy-union={busy/1000:.1f}ms ({busy/iters:.0f}us/iter)")
    print(f"GPU BUSY {100*busy/span:.1f}%   IDLE {100*idle/span:.1f}% = {idle/1000:.1f}ms ({idle/iters:.0f}us/iter)")

    # --- per-iteration steady-state idle (segment by cudaGraphLaunch ts; robust median) ---
    gl = sorted(t0 for t0,_ in host.get("cudaGraphLaunch", []))
    if len(gl) > 3:
        import statistics as st
        # build a sorted busy-interval list once for fast per-window intersection
        per_iter = []
        mi = 0
        for k in range(len(gl)-1):
            w0, w1 = gl[k], gl[k+1]
            wlen = w1 - w0
            # busy within [w0,w1] from merged intervals (merged is sorted, disjoint)
            b = 0.0
            for a,bb in merged:
                if bb <= w0: continue
                if a >= w1: break
                b += min(bb,w1) - max(a,w0)
            per_iter.append((wlen, wlen-b))
        durs = sorted(x[0] for x in per_iter)
        med_len = st.median(durs)
        # steady iters = those near the median length (exclude inter-round long-idle iters)
        steady = [(L,I) for (L,I) in per_iter if L <= 1.5*med_len]
        med_iter = st.median([L for L,_ in steady])
        med_idle = st.median([I for _,I in steady])
        print(f"\n=== per-ITER steady-state (segmented by cudaGraphLaunch; {len(steady)}/{len(per_iter)} iters within 1.5x median len) ===")
        print(f"median iter span={med_iter:.0f}us  median GPU-idle/iter={med_idle:.0f}us ({100*med_idle/med_iter:.1f}% of the step)")
        print(f"(the other {len(per_iter)-len(steady)} iters are inter-round capture idle, excluded)")
    print(f"\n=== per-GPU-stream busy (tid: us/iter, kernels/iter) ===")
    for tid,(c,d) in sorted(per_stream.items(), key=lambda x:-x[1][1])[:6]:
        print(f"  tid {tid:>7}: {d/iters:>8.0f}us/iter  {c/iters:>6.0f} kernels/iter")
    # idle gaps between merged busy intervals
    gaps = []
    for i in range(1, len(merged)):
        g0, g1 = merged[i-1][1], merged[i][0]
        if g1 - g0 > 1.0:  # >1us
            gaps.append((g1-g0, g0, g1))
    gaps.sort(reverse=True)
    print(f"\n=== top 15 GPU idle gaps (us) + host op overlapping each ===")
    sync = host.get("cudaEventSynchronize", []) + host.get("cudaStreamSynchronize", [])
    glaunch = host.get("cudaGraphLaunch", [])
    for sz, g0, g1 in gaps[:15]:
        # which host op overlaps this gap?
        tag = ""
        for s0,s1 in sync:
            if s0 < g1 and s1 > g0: tag = "<- cudaEventSync/StreamSync"; break
        if not tag:
            for s0,s1 in glaunch:
                if s0 < g1 and s1 > g0: tag = "<- cudaGraphLaunch"; break
        print(f"  gap {sz:>8.1f}us  {tag}")
    total_gap = sum(g[0] for g in gaps)
    print(f"\ntotal idle-gap time={total_gap/1000:.1f}ms ({total_gap/iters:.0f}us/iter) across {len(gaps)} gaps")
    # how much idle coincides with a host sync?
    sync_idle = 0.0
    for sz,g0,g1 in gaps:
        for s0,s1 in sync:
            if s0 < g1 and s1 > g0: sync_idle += sz; break
    print(f"idle overlapping cudaEventSync/StreamSync = {sync_idle/1000:.1f}ms ({sync_idle/iters:.0f}us/iter)  <- recoverable if the sync is removable")

if __name__ == "__main__": main()

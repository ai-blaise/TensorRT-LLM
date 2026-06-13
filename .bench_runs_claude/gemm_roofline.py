#!/usr/bin/env python3
"""Turn the nvfp4_gemm microbench latencies into a memory-roofline view:
achieved HBM bandwidth + %SOL vs B200 ~8 TB/s, for the decode GEMM shapes at a
given M. Answers 'are these GEMMs near memory-SOL (irreducible) or underutilized
(recoverable)?'. Reads the microbench stdout table from a file arg (or stdin)."""
import sys, re

PEAK_TBs = 8.0  # B200 HBM3e ~8 TB/s
# (name, N, K) — must match the microbench SHAPES
SHAPES = {
 "q_a_proj":(1536,7168),"q_b_proj":(24576,1536),"kv_a_proj_mqa":(576,7168),
 "kv_b_proj":(32768,512),"o_proj":(7168,16384),"dense_gate_up_full":(36864,7168),
 "dense_gate_up_tp4":(9216,7168),"dense_down_tp4":(7168,4608),
 "shared_gate_up":(4096,7168),"shared_down":(7168,2048),
}
BACKENDS=["cutlass","cublaslt","cuda_core"]  # matches microbench_fast (in-pool); bestALL/win%/reldiff trail

def bytes_moved(m,n,k):
    w = n*k*0.5 + (n*k/16)*1.0      # fp4 weight + UE4M3 block scales (1B / 16 elts)
    a = m*k*0.5 + (m*k/16)*1.0      # fp4 act + scales
    o = m*n*2.0                     # bf16 out
    return w+a+o

def main():
    txt = open(sys.argv[1]).read() if len(sys.argv)>1 else sys.stdin.read()
    rows=[]
    for ln in txt.splitlines():
        mt=re.match(r'\s+([a-z_]+)\s+(\d+)\s+(.*)', ln)
        if not mt: continue
        name,m=mt.group(1),int(mt.group(2))
        if name not in SHAPES: continue
        cells=mt.group(3).split()
        # first 4 numeric-or-'--' cells are the backend times
        bt={}
        for i,b in enumerate(BACKENDS):
            try: bt[b]=float(cells[i])
            except: pass
        rows.append((name,m,bt))
    print(f"{'shape':18s}{'M':>4s}{'best_us':>9s}{'best_be':>10s}{'MB_wt':>8s}{'GB/s':>8s}{'%SOL':>7s}")
    for name,m,bt in rows:
        if not bt: continue
        n,k=SHAPES[name]
        best_be=min(bt,key=bt.get); us=bt[best_be]
        gbs=bytes_moved(m,n,k)/(us*1e-6)/1e9
        sol=100*gbs/(PEAK_TBs*1000)
        wt_mb=(n*k*0.5+(n*k/16))/1e6
        flag=" <-- underutilized" if (m>=8 and sol<40) else (" near-SOL" if sol>=65 else "")
        print(f"{name:18s}{m:>4d}{us:>9.2f}{best_be:>10s}{wt_mb:>8.1f}{gbs:>8.0f}{sol:>6.0f}%{flag}")

if __name__=="__main__": main()

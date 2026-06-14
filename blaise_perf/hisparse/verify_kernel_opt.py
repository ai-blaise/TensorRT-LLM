#!/usr/bin/env python3
"""ORCHESTRATOR independent verification of the hot-read kernel optimization.
Loads a specific narrow .so (BEFORE or AFTER) via --library (argv[1]), runs the
op on FIXED-seed inputs, prints: dense-ref cosine (correctness), a bit-exact
output SIGNATURE (sum + first 8 vals), and CUDA-event timing. Run for BEFORE and
AFTER and compare: SIG must be identical (bit-exact), AFTER ~3.4x faster."""
import sys, statistics, torch
torch.ops.load_library(sys.argv[1])  # narrow th_hisparse_smoke .so (no import tensorrt_llm)

H_Q, D_QK, D_V = 128, 576, 512
TPB, KLR, QK_ROPE, KB = 64, 512, 64, 2
PACKED = TPB*(KLR*KB//8) + TPB*2*(KLR//128)*2 + TPB*QK_ROPE

def cos(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm()*b.norm() + 1e-12))

dev = torch.device("cuda:0"); sm = 1.0/(D_QK**0.5); NB = 32
g = torch.Generator(device="cpu").manual_seed(7)
lat = (torch.randn(NB, TPB, D_QK, generator=g)*0.5).to(dev, torch.float16)
hot = torch.zeros(NB, PACKED, dtype=torch.uint8, device=dev)
for b in range(NB):
    torch.ops.trtllm.mla_bdr_write_kvarn_record(lat[b], hot, b, KB, KLR, QK_ROPE)
torch.cuda.synchronize()
hot = hot.unsqueeze(0)
# dequant the kernel sees, via native BDR reader
sf = TPB
idxd = torch.empty(NB, TPB, dtype=torch.int32, device=dev)
for b in range(NB):
    for t in range(TPB):
        idxd[b, t] = b*sf + t
tl = torch.full((NB,), TPB, dtype=torch.int32, device=dev)
rs0 = torch.zeros((NB,), dtype=torch.uint8, device=dev)
dq, _ = torch.ops.trtllm.hisparse_read_kvarn_hot_bdr(hot, idxd, tl, rs0, 0, TPB, KB, KLR, QK_ROPE)
dq = dq.float(); torch.cuda.synchronize()

B, ns = 16, 1024
g2 = torch.Generator(device="cpu").manual_seed(11)
q = (torch.randn(B, 1, H_Q, D_QK, generator=g2)*0.3).to(dev, torch.bfloat16)
idx = torch.empty(B, 1, ns, dtype=torch.int32, device=dev)
seldq = torch.empty(B, ns, D_QK, device=dev)
for b in range(B):
    blk = torch.randint(0, NB, (ns,), generator=g2)
    tok = torch.randint(0, TPB, (ns,), generator=g2)
    idx[b, 0] = (blk*TPB + tok).to(torch.int32).to(dev)
    seldq[b] = dq[blk, tok, :]
rs = torch.zeros((B,), dtype=torch.uint8, device=dev)

def run():
    o, l, m, s = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        q, hot, idx, rs, None, None, 0, TPB, TPB, KB, KLR, QK_ROPE, sm)
    return o

o = run(); torch.cuda.synchronize()
qf = q.float(); K = seldq; V = seldq[..., :D_V]
sc = sm*torch.einsum("bhd,bnd->bhn", qf[:, 0], K)
mx = sc.max(-1, keepdim=True).values
w = torch.exp(sc-mx); wn = w/w.sum(-1, keepdim=True)
ref = torch.einsum("bhn,bnd->bhd", wn, V).unsqueeze(1)
print("dense-ref cosine =", round(cos(o.float(), ref), 6))
print("SIG sum=%.10e  head8=%s" % (o.float().double().sum().item(),
      [round(x, 6) for x in o.float().reshape(-1)[:8].tolist()]))
st = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
en = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
for _ in range(3): run()
torch.cuda.synchronize()
for i in range(20):
    st[i].record(); run(); en[i].record()
torch.cuda.synchronize()
ms = [s.elapsed_time(e) for s, e in zip(st, en)]
print("timing: ms/call mean=%.3f  ms/row=%.4f" % (statistics.mean(ms), statistics.mean(ms)/B))

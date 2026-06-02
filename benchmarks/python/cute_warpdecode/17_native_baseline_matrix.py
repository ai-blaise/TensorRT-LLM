"""Full native NVFP4 MoE pipeline (FP4BlockScaleMoERunner.forward) — graph-captured production baseline.
Routing -> sort -> grouped GEMM FC1 -> SwiGLU -> grouped GEMM FC2 -> finalize. The real baseline."""
import os, torch, numpy as np
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner, ActType_TrtllmGen
HIDDEN, INTERMEDIATE = 7168, 2048; TK=8; SV=16
device=torch.device("cuda")

def make_runner(LE):
    return FP4BlockScaleMoERunner(128, TK, 8, 4, INTERMEDIATE, 0, LE, None, 2, True, ActType_TrtllmGen.SwiGlu.value, tune_max_num_tokens=8192, use_dp=False)

def weights(LE):
    return (torch.randint(0,256,(LE,2*INTERMEDIATE,HIDDEN//2),device=device,dtype=torch.uint8),
            torch.randint(0,256,(LE,2*INTERMEDIATE,HIDDEN//SV),device=device,dtype=torch.uint8).view(torch.float8_e4m3fn),
            torch.randint(0,256,(LE,HIDDEN,INTERMEDIATE//2),device=device,dtype=torch.uint8),
            torch.randint(0,256,(LE,HIDDEN,INTERMEDIATE//SV),device=device,dtype=torch.uint8).view(torch.float8_e4m3fn))

def routing(tokens, slot, LE, seed):
    rng=np.random.default_rng(seed)
    experts=rng.choice(LE, size=min(slot,LE), replace=False)
    return torch.from_numpy(rng.choice(experts, size=(tokens,TK), replace=True)).to(device,dtype=torch.int32).contiguous()

def mk(tokens):
    return {"x":torch.randint(0,256,(tokens,HIDDEN//2),device=device,dtype=torch.uint8),
            "x_sf":torch.randint(0,256,(tokens,HIDDEN//SV),device=device,dtype=torch.uint8).view(torch.float8_e4m3fn).flatten(),
            "weights":torch.full((tokens,TK),1.0/TK,device=device,dtype=torch.bfloat16)}

def fwd(runner, d, ids, w13,w13s,w2,w2s, LE, tac):
    o1=torch.ones((LE,),device=device,dtype=torch.float32)
    return runner.forward([None,None,d["x"],d["x_sf"],w13,w13s,None,None,None,None,w2,w2s,None,o1,o1,o1,d["weights"],ids], tactic=tac)

def valid_tactic(runner, d, ids, w13,w13s,w2,w2s, LE):
    o1=torch.ones((LE,),device=device,dtype=torch.float32)
    args=[None,None,d["x"],d["x_sf"],w13,w13s,None,None,None,None,w2,w2s,None,o1,o1,o1,d["weights"],ids]
    try:
        tacs=runner.get_valid_tactics(args) if hasattr(runner,"get_valid_tactics") else None
    except Exception: tacs=None
    # try a few candidate tactics, return first that runs
    cands=[[32,36],[16,36],[16,52],[8,36],[0],[0,0],[1]]
    for c in cands:
        try:
            runner.forward(args, tactic=c); torch.cuda.synchronize(); return c
        except Exception as ex:
            last=repr(ex)[:120]; continue
    print("   tactic search last err:", last)
    return None

def graph_time(callfn, nrep=20, it=50):
    for _ in range(10): callfn(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(g):
        for _ in range(nrep): callfn(torch.cuda.current_stream())
    torch.cuda.synchronize(); e1=torch.cuda.Event(True); e2=torch.cuda.Event(True); e1.record()
    for _ in range(it): g.replay()
    e2.record(); torch.cuda.synchronize(); return e1.elapsed_time(e2)/it/nrep*1000

print("=== FULL native NVFP4 MoE pipeline (runner.forward) — graph-captured ===")
for (G,LE) in [(8,16),(4,32),(2,64)]:
    w13,w13s,w2,w2s=weights(LE); runner=make_runner(LE)
    for conc in [16,32]:
        ids=routing(conc, LE, LE, 42); d=mk(conc)
        tac=valid_tactic(runner,d,ids,w13,w13s,w2,w2s,LE)
        if tac is None:
            print(f"G={G} conc={conc:2d} (LE={LE:2d}): no valid tactic found"); continue
        t=graph_time(lambda st: fwd(runner,d,ids,w13,w13s,w2,w2s,LE,tac))
        print(f"G={G} conc={conc:2d} (LE={LE:2d}): native full-pipeline = {t:7.2f}us  (tactic={tac})")

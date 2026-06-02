"""3-way matched per-rank decode comparison: native vs WarpDecode-default vs WarpDecode-2CTA(autotuned).
The cute_dsl FC1/FC2 ops pick (mma_tiler, cluster_shape) via AutoTuner.choose_one; the 2-CTA/256x256
tactic is in that grid. Running the ops under `with autotune()` lets the tuner profile the full grid
(incl 2-CTA) and cache the best tactic for the decode shape -> WarpDecode's optimal config. Without
autotune the op uses an UNtuned default tactic, which sandbags WarpDecode. Identical per-rank shape
(experts=128/G, tokens=conc*8/G); a2a is common to both paths and excluded (warp decode does NOT
remove inter-GPU comm -- per cursor.com/blog/warp-decode it removes only LOCAL bookkeeping)."""
import os, json, torch, pathlib
import tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops
import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner, ActType_TrtllmGen
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._torch.autotuner import autotune
HIDDEN, INTERMEDIATE = 7168, 2048
NE, TK, NG, TG, SV, DSR = 128, 8, 8, 4, 16, 2
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
dev = torch.device("cuda")

def graph_time(fn, tune=False, it=80, rp=5, warm=15):
    if tune:
        with autotune():
            for _ in range(10): fn()
        torch.cuda.synchronize()
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize()
    for _ in range(warm): g.replay()
    torch.cuda.synchronize()
    v = []
    for _ in range(rp):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
        for _ in range(it): g.replay()
        e1.record(); torch.cuda.synchronize(); v.append(e0.elapsed_time(e1) / it)
    del g; return min(v) * 1000

def run_shape(LE, ntok):
    w13 = torch.randint(0, 256, (LE, 2 * INTERMEDIATE, HIDDEN // 2), device=dev, dtype=torch.uint8)
    w13sf = torch.randint(1, 8, (LE, 2 * INTERMEDIATE, HIDDEN // SV), device=dev, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (LE, HIDDEN, INTERMEDIATE // 2), device=dev, dtype=torch.uint8)
    w2sf = torch.randint(1, 8, (LE, HIDDEN, INTERMEDIATE // SV), device=dev, dtype=torch.uint8)
    a1 = torch.ones((LE,), device=dev, dtype=torch.float32); gsf = torch.ones((1,), device=dev, dtype=torch.float32)
    runner = FP4BlockScaleMoERunner(NE, TK, NG, TG, INTERMEDIATE, 0, LE, None, DSR, True, ActType_TrtllmGen.SwiGlu.value, tune_max_num_tokens=8192, use_dp=False)
    TB = {1: [64, 6], 2: [64, 4], 4: [64, 4], 8: [32, 36], 16: [16, 52], 32: [32, 36], 64: [32, 36], 128: [32, 36], 256: [32, 36]}
    def tac(n):
        b = 1
        for k in sorted(TB):
            if n <= k: b = k; break
        else: b = max(TB)
        return TB[b]
    x4 = torch.randint(0, 256, (ntok, HIDDEN // 2), device=dev, dtype=torch.uint8)
    xsf = torch.randint(1, 8, (ntok, HIDDEN // SV), device=dev, dtype=torch.uint8)
    ids = (torch.arange(ntok * TK, device=dev, dtype=torch.int32).reshape(ntok, TK) % LE).contiguous()
    wf32 = torch.full((ntok, TK), 1.0 / TK, device=dev, dtype=torch.float32); wbf = wf32.to(torch.bfloat16)
    xsf_f = xsf.view(torch.float8_e4m3fn).flatten()
    def native():
        return runner.forward([None, None, x4, xsf_f, w13, w13sf.view(torch.float8_e4m3fn), None, None, None, None, w2, w2sf.view(torch.float8_e4m3fn), None, a1, a1, a1, wbf, ids], tactic=tac(ntok))[0]
    meta = torch.ops.trtllm.moe_sort(token_selected_experts=ids, token_final_scales=wf32, num_experts=NE, top_k=TK, local_expert_offset=0, local_num_experts=LE, tile_tokens_dim=128)
    t2e, t2lim, e2p, p2e, tot, nt = meta
    out = torch.zeros(ntok, HIDDEN, device=dev, dtype=torch.bfloat16)
    def wd():
        h, hsf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
            input=x4.view(torch.float4_e2m1fn_x2), weight=w13.view(torch.float4_e2m1fn_x2), input_scale=xsf.view(torch.uint8),
            weight_scale=w13sf.view(torch.uint8), alpha=a1, tile_idx_to_group_idx=t2e, tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e, num_non_exiting_tiles=nt, global_sf=gsf, num_experts=NE, top_k=TK,
            num_local_experts=LE, local_expert_offset=0, tile_size=128, scaling_vector_size=SV, activation_type=int(ActivationType.Swiglu))
        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
            input=h.view(torch.float4_e2m1fn_x2), weight=[w2.view(torch.float4_e2m1fn_x2)], input_scale=hsf.view(torch.uint8),
            weight_scale=[w2sf.view(torch.uint8)], alpha=[a1], output=out, tile_idx_to_group_idx=t2e, tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e, num_non_exiting_tiles=nt, token_final_scales=wf32, num_experts=NE, top_k=TK,
            num_local_experts=LE, local_expert_offset=0, tile_size=128, output_dtype=torch.bfloat16)
        return out
    nat = graph_time(native)
    wd_def = graph_time(wd)              # empty tuner cache -> UNtuned default tactic
    wd_tuned = graph_time(wd, tune=True)  # AutoTuner profiles full grid (incl 2-CTA/256x256) -> best
    del w13, w13sf, w2, w2sf, runner; torch.cuda.empty_cache()
    return nat, wd_def, wd_tuned

rows = []
print(f"{'GPUs':>4} {'conc':>4} {'exp/rk':>6} {'tok/rk':>6} {'native':>9} {'WD_default':>10} {'WD_2CTAtuned':>12} {'nat/def':>8} {'nat/tuned':>9} {'tuned/def':>9}", flush=True)
for G in [2, 4, 8]:
    LE = NE // G
    for conc in [16, 32]:
        ntok = max(conc * TK // G, 1)
        nat, wd_def, wd_tuned = run_shape(LE, ntok)
        r = {"G": G, "conc": conc, "LE": LE, "ntok": ntok, "native_us": round(nat, 2), "wd_default_us": round(wd_def, 2),
             "wd_2cta_tuned_us": round(wd_tuned, 2), "sp_nat_vs_def": round(nat / wd_def, 3), "sp_nat_vs_tuned": round(nat / wd_tuned, 3),
             "sp_tuned_vs_def": round(wd_def / wd_tuned, 3)}
        rows.append(r)
        print(f"{G:>4} {conc:>4} {LE:>6} {ntok:>6} {nat:>9.2f} {wd_def:>10.2f} {wd_tuned:>12.2f} {nat/wd_def:>7.3f}x {nat/wd_tuned:>8.3f}x {wd_def/wd_tuned:>8.3f}x", flush=True)
pathlib.Path('/home/spencer/work/benches/artifacts').mkdir(parents=True, exist_ok=True)
pathlib.Path('/home/spencer/work/benches/artifacts/wd_2cta_compare.json').write_text(json.dumps(rows, indent=2))
print("\nWD_default     = cute_dsl FC1/FC2 with UNtuned default tactic (= the first matched run).", flush=True)
print("WD_2CTAtuned   = same ops under AutoTuner over full mma_tiler x cluster grid incl 2-CTA/256x256 -> best.", flush=True)
print("native         = FP4BlockScaleMoERunner (trtllm_gen) with hand-tuned tactic.", flush=True)
print("MoE decode is context-independent: each (GPU,conc) row holds across all contexts 1k-128k.", flush=True)

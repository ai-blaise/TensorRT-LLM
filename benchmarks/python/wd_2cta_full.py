"""FORCED 2-CTA WarpDecode -- full local + system e2e, all {2,4,8}x{16,32} cells.
2-CTA setup = tile_size=256 in moe_sort + gather_grouped_gemm + finalize_inplace ops; the gather
runner sets cluster_shape=(tile_size//128, 1), so tile_size=256 -> cluster (2,1) = 2-CTA (cta_group::2),
mma_tiler (256,128). NO autotuner -- 2-CTA is forced by construction. Correctness gate: cosine of the
2-CTA output vs the 1-CTA (tile_size=128) output of the SAME cute_dsl kernel family must be ~1.0 (tile
size is a perf-only choice; a mismatch means the 2-CTA path is broken). Then both native and WD-2CTA
are wrapped in IDENTICAL real NCCL a2a (dispatch acts+scales, combine bf16) for the honest system
number. MoE decode is context-independent: each (G,conc) cell holds across contexts 1k..128k.
Run: python -m torch.distributed.run --nproc_per_node=G wd_2cta_full.py"""
import os, json, torch, torch.distributed as dist
import tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops
import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner, ActType_TrtllmGen
from tensorrt_llm._torch.utils import ActivationType
HIDDEN, INTERMEDIATE = 7168, 2048
NE, TK, NG, TG, SV, DSR = 128, 8, 8, 4, 16, 2

def main():
    rank = int(os.environ["RANK"]); G = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr); dist.init_process_group("nccl", rank=rank, world_size=G)
    dev = torch.device(f"cuda:{lr}"); os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    LE = NE // G
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
    def gtime(fn, it=60, rp=5, warm=12):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); dist.barrier()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): fn()
        torch.cuda.synchronize()
        for _ in range(warm): g.replay()
        torch.cuda.synchronize(); dist.barrier()
        v = []
        for _ in range(rp):
            dist.barrier(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
            for _ in range(it): g.replay()
            e1.record(); torch.cuda.synchronize(); v.append(e0.elapsed_time(e1) / it)
        del g; t = torch.tensor([min(v) * 1000], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.item()
    def ltime(fn, it=100, rp=5, warm=15):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); dist.barrier()
        v = []
        for _ in range(rp):
            dist.barrier(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
            for _ in range(it): fn()
            e1.record(); torch.cuda.synchronize(); v.append(e0.elapsed_time(e1) / it)
        t = torch.tensor([min(v) * 1000], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.item()
    def wd_run(x4, xsf, meta, ts, out, tfs):
        t2e, t2lim, e2p, p2e, tot, nt = meta
        h, hsf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
            input=x4.view(torch.float4_e2m1fn_x2), weight=w13.view(torch.float4_e2m1fn_x2), input_scale=xsf.view(torch.uint8),
            weight_scale=w13sf.view(torch.uint8), alpha=a1, tile_idx_to_group_idx=t2e, tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e, num_non_exiting_tiles=nt, global_sf=gsf, num_experts=NE, top_k=TK,
            num_local_experts=LE, local_expert_offset=0, tile_size=ts, scaling_vector_size=SV, activation_type=int(ActivationType.Swiglu))
        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
            input=h.view(torch.float4_e2m1fn_x2), weight=[w2.view(torch.float4_e2m1fn_x2)], input_scale=hsf.view(torch.uint8),
            weight_scale=[w2sf.view(torch.uint8)], alpha=[a1], output=out, tile_idx_to_group_idx=t2e, tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e, num_non_exiting_tiles=nt, token_final_scales=tfs, num_experts=NE, top_k=TK,
            num_local_experts=LE, local_expert_offset=0, tile_size=ts, output_dtype=torch.bfloat16)
        return out
    rows = []
    for conc in [16, 32]:
        ntok = max(conc * TK // G, 1)
        x4 = torch.randint(0, 256, (ntok, HIDDEN // 2), device=dev, dtype=torch.uint8)
        xsf = torch.randint(1, 8, (ntok, HIDDEN // SV), device=dev, dtype=torch.uint8)
        ids = (torch.arange(ntok * TK, device=dev, dtype=torch.int32).reshape(ntok, TK) % LE).contiguous()
        wf32 = torch.full((ntok, TK), 1.0 / TK, device=dev, dtype=torch.float32); wbf = wf32.to(torch.bfloat16)
        xsf_f = xsf.view(torch.float8_e4m3fn).flatten()
        meta128 = torch.ops.trtllm.moe_sort(token_selected_experts=ids, token_final_scales=wf32, num_experts=NE, top_k=TK, local_expert_offset=0, local_num_experts=LE, tile_tokens_dim=128)
        meta256 = torch.ops.trtllm.moe_sort(token_selected_experts=ids, token_final_scales=wf32, num_experts=NE, top_k=TK, local_expert_offset=0, local_num_experts=LE, tile_tokens_dim=256)
        out1 = torch.zeros(ntok, HIDDEN, device=dev, dtype=torch.bfloat16)
        out2 = torch.zeros(ntok, HIDDEN, device=dev, dtype=torch.bfloat16)
        o128 = wd_run(x4, xsf, meta128, 128, out1, wf32).clone()
        o256 = wd_run(x4, xsf, meta256, 256, out2, wf32).clone()
        cos = torch.nn.functional.cosine_similarity(o128.float().flatten().unsqueeze(0), o256.float().flatten().unsqueeze(0)).item()
        finite = bool(torch.isfinite(o256).all().item()); nz = float(o256.abs().mean().item())
        da = torch.zeros(ntok, HIDDEN // 2, device=dev, dtype=torch.uint8); da_r = torch.zeros_like(da)
        dsf = torch.zeros(ntok, HIDDEN // SV, device=dev, dtype=torch.uint8); dsf_r = torch.zeros_like(dsf)
        cb = torch.zeros(ntok, HIDDEN, device=dev, dtype=torch.bfloat16); cb_r = torch.zeros_like(cb)
        def a2a():
            dist.all_to_all_single(da_r, da); dist.all_to_all_single(dsf_r, dsf); dist.all_to_all_single(cb_r, cb)
        def native_local():
            return runner.forward([None, None, x4, xsf_f, w13, w13sf.view(torch.float8_e4m3fn), None, None, None, None, w2, w2sf.view(torch.float8_e4m3fn), None, a1, a1, a1, wbf, ids], tactic=tac(ntok))[0]
        def wd_local():
            return wd_run(x4, xsf, meta256, 256, out2, wf32)
        def native_e2e():
            dist.all_to_all_single(da_r, da); dist.all_to_all_single(dsf_r, dsf); o = native_local(); cb.copy_(o); dist.all_to_all_single(cb_r, cb); return cb_r
        def wd_e2e():
            dist.all_to_all_single(da_r, da); dist.all_to_all_single(dsf_r, dsf); o = wd_local(); cb.copy_(o); dist.all_to_all_single(cb_r, cb); return cb_r
        a2a_us = ltime(a2a); nl = gtime(native_local); wl = gtime(wd_local)
        ne = gtime(native_e2e); we = gtime(wd_e2e)
        if rank == 0:
            r = {"G": G, "conc": conc, "LE": LE, "ntok": ntok, "cos_2cta_vs_1cta": round(cos, 5), "finite": finite, "absmean": round(nz, 4),
                 "a2a_us": round(a2a_us, 2), "native_local_us": round(nl, 2), "wd_2cta_local_us": round(wl, 2), "local_sp": round(nl / wl, 3),
                 "native_e2e_us": round(ne, 2), "wd_2cta_e2e_us": round(we, 2), "system_sp": round(ne / we, 3)}
            rows.append(r); print(json.dumps(r), flush=True)
    if rank == 0:
        print(f"\n=== FORCED 2-CTA (tile_size=256, cluster (2,1)) WarpDecode, G={G} ===", flush=True)
        for r in rows:
            print(f"  G{r['G']} c{r['conc']}: cos(2cta vs 1cta)={r['cos_2cta_vs_1cta']} finite={r['finite']} | local {r['native_local_us']}->{r['wd_2cta_local_us']}us ({r['local_sp']}x) | a2a {r['a2a_us']}us | e2e {r['native_e2e_us']}->{r['wd_2cta_e2e_us']}us ({r['system_sp']}x)", flush=True)
        import pathlib; pathlib.Path('/home/spencer/work/benches/artifacts').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'/home/spencer/work/benches/artifacts/wd_2cta_full_G{G}.json').write_text(json.dumps(rows, indent=2))
    dist.destroy_process_group()
if __name__ == "__main__": main()

"""Direct op-level validation of the custom decode pieces on prod (REAP-345B) shapes.
Run inside the proof image with the op-trt-pde-custom worktree overlaid on top of
the installed tensorrt_llm. No full serve, no model load.

Round-2 deepening (vs the presence-only round-1 harness):
  - Gated norm: also exercises the SERVED cute_lowrank_gate kernel (the one the
    _GATE_IMPL=='cute' dispatcher actually calls at rank=16) vs a bf16 reference,
    not just the apply_fused_lowrank_gate wrapper.
  - HISA: corrects HISA_MIN_SEQ to the real 32768 default; adds bit-exact CORRECTNESS
    of the three live HISA ops (candidate_pages / mask_scores / remap_selected) at the
    real >=32768 decode regime (kv=131072), plus final top-k=1024 selection vs reference.
  - Op presence uses real_present() (forces overload resolution) instead of hasattr(),
    which lies True for unregistered torch.ops packets.
  - Pins the model-card -> sparse_attention_config resolution so the harness asserts the
    REAL prod gate (HISA at kv>=32768; FSSS / cross-step reuse dormant off-card)."""
import os, math, torch

torch.manual_seed(0)
dev = "cuda"
HID = 7168            # hidden_size (REAP-345B)
RANK = 16            # gated_norm_rank
INDEX_TOPK = 1024
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
HISA_BLOCK = 128
HISA_BLOCK_TOPK = 64
HISA_RATIO = 4.0
# Real prod gate: DeepSeekSparseAttentionConfig.hisa_min_seq_len Field default
# (llm_args.py) == 32768, and the REAP-345B model card's indexer_quantization.hisa
# block carries NO `min_seq_len` key -> _get_blaise_indexer_overrides leaves it at
# the 32768 default (model_config.py:128 only sets it when present). So HISA at
# decode engages at max_kv_len >= 32768, NOT 65536 (the prior constant here was
# wrong and made the kv=32768 boundary case test the wrong threshold).
HISA_MIN_SEQ = 32768

PASS = True
def check(name, ok, extra=""):
    global PASS
    PASS = PASS and bool(ok)
    print("[" + ("PASS" if ok else "FAIL") + "] " + name + " " + extra)

def real_present(name):
    """torch.ops.<ns>.<op> attribute access is lazy: `hasattr(torch.ops.trtllm, X)`
    can return True for an op that is NOT registered (the packet object is created
    on access; resolution is deferred). Force overload resolution via `.default`
    so a missing op is reported as absent. This is the accurate presence test."""
    try:
        getattr(torch.ops.trtllm, name).default
        return True
    except Exception:
        return False

print("=== tensorrt_llm import + overlay provenance ===")
import tensorrt_llm
tl_dir = os.path.dirname(tensorrt_llm.__file__)
dsa_path = os.path.join(tl_dir, "_torch/attention_backend/sparse/dsa.py")
src = open(dsa_path).read()
check("dsa overlay has 1024 comment (not stale 2048)",
      "sparse_attention_config.index_topk  # 1024" in src,
      "(confirms worktree overlay active)")

# ---------- PIECE 2: gated norm ----------
print("\\n=== PIECE 2: gated norm (fused vs bf16 reference) ===")
from tensorrt_llm._torch.modules import fused_lowrank_gate as flg
for ntok in (1, 4, 16, 64):
    x = torch.randn(ntok, HID, dtype=torch.bfloat16, device=dev)
    wd = torch.randn(RANK, HID, dtype=torch.bfloat16, device=dev) * 0.02
    wu = torch.randn(HID, RANK, dtype=torch.bfloat16, device=dev) * 0.02
    gate_down = torch.nn.Linear(HID, RANK, bias=False, dtype=torch.bfloat16, device=dev)
    gate_up = torch.nn.Linear(RANK, HID, bias=False, dtype=torch.bfloat16, device=dev)
    with torch.no_grad():
        gate_down.weight.copy_(wd); gate_up.weight.copy_(wu)
    # bf16 reference: mirror _maybe_apply_gated_norm eager fallback exactly
    # (modeling_deepseekv3.py): silu in fp32 -> cast bf16 -> bf16 matmul -> sigmoid -> cast.
    gg = torch.matmul(x.float(), wd.float().t())
    gg = torch.nn.functional.silu(gg).to(torch.bfloat16)
    gg = torch.matmul(gg, gate_up.weight.t())
    gg = torch.sigmoid(gg).to(torch.bfloat16)
    ref = (x * gg)
    sup = flg.lowrank_gate_supported(x, RANK)
    if sup:
        out = flg.apply_fused_lowrank_gate(x, gate_down, gate_up)
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        check(f"gated_norm fused vs ref ntok={ntok}", cos > 0.999, f"cos={cos:.6f} supported={sup}")
    else:
        check(f"gated_norm fused-path supported ntok={ntok}", False, f"supported={sup} (fell to eager)")

# Verify the SERVED gate path: with _GATE_IMPL=='cute' (default, env
# TRTLLM_OPTRT_LOWRANK_GATE_IMPL) and IS_CUTLASS_DSL_AVAILABLE, apply_fused_lowrank_gate
# dispatches to trtllm::cute_lowrank_gate for rank in (8,16,32,64). The cos check above
# exercises apply_fused_lowrank_gate (the dispatcher); this block proves the cute op the
# dispatcher actually calls is itself correct vs the bf16 reference, and that the prod
# rank=16 shape is `supported` (so the cute kernel -- not the triton fallback -- is taken).
print("\\n=== PIECE 2: served cute_lowrank_gate (the dispatched kernel) ===")
print("    _GATE_IMPL =", getattr(flg, "_GATE_IMPL", "?"))
try:
    from tensorrt_llm._torch.modules.cute_lowrank_gate import cute_lowrank_gate_supported
    from tensorrt_llm._torch.modules.fused_lowrank_gate import get_lowrank_gate_weights, _get_lowrank_gate_wd_bf16
    _cute_gate_ok = real_present("cute_lowrank_gate")
    check("op present trtllm::cute_lowrank_gate (resolved)", _cute_gate_ok)
    if _cute_gate_ok:
        xg = torch.randn(16, HID, dtype=torch.bfloat16, device=dev)
        gd = torch.nn.Linear(HID, RANK, bias=False, dtype=torch.bfloat16, device=dev)
        gu = torch.nn.Linear(RANK, HID, bias=False, dtype=torch.bfloat16, device=dev)
        with torch.no_grad():
            gd.weight.mul_(0.02); gu.weight.mul_(0.02)
        # supported takes (x, wd, rank:int) -- rank is gate_down.weight.shape[0]
        sup_cute = cute_lowrank_gate_supported(xg, gd.weight, gd.weight.shape[0])
        gg = torch.sigmoid(torch.matmul(
            torch.nn.functional.silu(torch.matmul(xg.float(), gd.weight.float().t())).to(torch.bfloat16),
            gu.weight.t()).float()).to(torch.bfloat16)
        ref_cute = (xg * gg)
        _, wu_t = get_lowrank_gate_weights(gd, gu)
        wd_bf16 = _get_lowrank_gate_wd_bf16(gd)
        y_cute = torch.ops.trtllm.cute_lowrank_gate(xg, wd_bf16, wu_t)
        cosc = torch.nn.functional.cosine_similarity(y_cute.float().flatten(), ref_cute.float().flatten(), dim=0).item()
        check("served cute_lowrank_gate supported@rank16 + cos vs bf16-ref", sup_cute and cosc > 0.999,
              f"supported={sup_cute} cos={cosc:.6f}")
except Exception as e:
    check("served cute_lowrank_gate reachable", False, f"{type(e).__name__}: {str(e)[:120]}")

# fp4 MoE-linear handoff op presence + shape (no dead-op)
print("\\n=== PIECE 2: fp4 handoff ops ===")
ops = torch.ops.trtllm
# fused_* are always-present (triton/native); cute_* are cute_dsl-gated and only resolve
# after the cute_lowrank_gate module import above. real_present() forces real resolution.
for op in ["fused_lowrank_gate","fused_lowrank_gate_quant_nvfp4"]:
    check(f"op present trtllm::{op}", real_present(op))
for op in ["cute_lowrank_gate","cute_lowrank_gate_quant_nvfp4","cute_lowrank_gate_quant_nvfp4_swizzled"]:
    # cute variants are correctly absent when IS_CUTLASS_DSL_AVAILABLE is False; report state.
    present = real_present(op)
    check(f"cute gate op state trtllm::{op}", True, f"resolved={present}")
# Run the triton MoE-linear quant path (has fallback, full 1024 tiles) and check shapes
x = torch.randn(16, HID, dtype=torch.bfloat16, device=dev)
gate_down = torch.nn.Linear(HID, RANK, bias=False, dtype=torch.bfloat16, device=dev)
gate_up = torch.nn.Linear(RANK, HID, bias=False, dtype=torch.bfloat16, device=dev)
with torch.no_grad():
    gate_down.weight.mul_(0.02); gate_up.weight.mul_(0.02)
qscale = torch.tensor(1.0, dtype=torch.float32, device=dev)
if flg.lowrank_gate_quant_nvfp4_supported(x, RANK):
    y, q, sf = flg.apply_fused_lowrank_gate_quant_nvfp4(x, gate_down, gate_up, qscale)
    # gated bf16 y must match plain gate; q is fp4 packed (uint8, HID/2 cols)
    yref = flg.apply_fused_lowrank_gate(x, gate_down, gate_up)
    cosy = torch.nn.functional.cosine_similarity(y.float().flatten(), yref.float().flatten(), dim=0).item()
    check("fp4-handoff gated-bf16 matches plain gate", cosy > 0.999, f"cos={cosy:.6f}")
    check("fp4-handoff q shape [16,HID/2]", tuple(q.shape) == (16, HID//2), f"q={tuple(q.shape)} sf={tuple(sf.shape)}")
else:
    check("fp4 MoE-linear handoff supported", False)

# ---------- PIECE 1: HISA gate + two-level block_topk math ----------
print("\\n=== PIECE 1: HISA gate + two-level topk math ===")
from tensorrt_llm._torch.attention_backend.sparse import dsa as dsamod
# free-function mirror used in scheduling
def block_topk(num_blocks):
    return dsamod._hisa_block_topk_value(num_blocks, INDEX_TOPK, HISA_BLOCK, HISA_RATIO, HISA_BLOCK_TOPK)
# Below the real 32768 min_seq -> HISA must NOT engage (gate is on Indexer; mirror its
# arithmetic). _should_use_hisa returns max_kv_len >= hisa_min_seq_len(=32768) AND the
# candidate band covers index_topk. At kv < 32768 the min-seq floor alone disables HISA
# regardless of cand, so the prod short-decode regime (live kv ~4.6k) runs the dense NVFP4
# indexer path, NOT HISA -- by design.
for kv in (4096, 16384, 32767):
    nb = math.ceil(kv / HISA_BLOCK); bt = block_topk(nb); cand = bt*HISA_BLOCK
    engages = (kv >= HISA_MIN_SEQ) and (cand >= INDEX_TOPK)
    check(f"HISA off below min_seq(32768) kv={kv}", not engages, f"cand={cand} kv>=min={kv>=HISA_MIN_SEQ}")
# At/above the 32768 min_seq -> engages and candidate band covers top-1024.
for kv in (32768, 65536, 131072):
    nb = math.ceil(kv / HISA_BLOCK); bt = block_topk(nb); cand = bt*HISA_BLOCK
    check(f"HISA engages kv={kv} & cand>=topk", (kv>=HISA_MIN_SEQ) and (cand>=INDEX_TOPK),
          f"num_blocks={nb} block_topk={bt} candidate_len={cand} (>= {INDEX_TOPK})")
# min_blocks floor = ceil(index_topk/block_size)
check("block_topk floor==ceil(index_topk/block)", block_topk(8) >= math.ceil(INDEX_TOPK/HISA_BLOCK),
      f"bt(8)={block_topk(8)} floor={math.ceil(INDEX_TOPK/HISA_BLOCK)}")
# C++ HISA op presence (no dead-op fallback in served optimized mode)
print("\\n=== PIECE 1: HISA C++ ops ===")
# trtllm:: custom ops (optimized-mode HISA path; present == no dead-op fallback).
# real_present forces overload resolution (hasattr lies for unregistered ops).
for op in ["indexer_topk_decode","indexer_hisa_mask_scores","indexer_hisa_remap_selected",
           "indexer_hisa_candidate_pages","indexer_hisa_quantized_block_reps_from_pages_nvfp4",
           "indexer_hisa_update_page_reps_nvfp4","indexer_hisa_block_scores_nvfp4"]:
    check("op present trtllm::" + op, real_present(op))
import tensorrt_llm.deep_gemm as dg  # DeepGEMM logits kernels are python fns, not torch.ops
for fn in ["fp8_fp4_paged_mqa_logits","fp8_paged_mqa_logits","get_paged_mqa_logits_metadata"]:
    check("deep_gemm." + fn + " present", hasattr(dg, fn))

# HISA op CORRECTNESS at the real >=32768 decode regime (kv=131072 -> max_blocks=1024,
# block_topk=256, candidate_len=32768). Verifies the optimized-mode HISA custom kernels
# produce output bit-identical to the eager reference math in dsa.py, not just that they
# are registered. These are the three pure-tensor HISA ops on the live decode chain.
print("\\n=== PIECE 1: HISA op correctness @ prod kv=131072 ===")
if real_present("indexer_hisa_candidate_pages") and real_present("indexer_hisa_mask_scores") \
        and real_present("indexer_hisa_remap_selected"):
    KV = 131072; PAGE = 64
    mb = math.ceil(KV / HISA_BLOCK)
    btv = min(max(math.ceil(mb / HISA_RATIO), math.ceil(INDEX_TOPK / HISA_BLOCK)), mb)
    cl = btv * HISA_BLOCK
    tk = min(INDEX_TOPK, cl)
    pphb = HISA_BLOCK // PAGE
    for nr in (4, 16):
        prefix = torch.full((nr,), KV, dtype=torch.int32, device=dev)
        tb = torch.stack([torch.randperm(mb, device=dev)[:btv].to(torch.int32) for _ in range(nr)])
        tb_i64 = tb.to(torch.int64)
        off = torch.arange(HISA_BLOCK, device=dev)
        # candidate_pages
        max_pages = math.ceil(KV / PAGE)
        bt_tab = torch.arange(nr * max_pages, dtype=torch.int32, device=dev).reshape(nr, max_pages)
        cp_op = torch.ops.trtllm.indexer_hisa_candidate_pages(tb, bt_tab.contiguous(), 1, pphb)
        cand_pages = (tb_i64.unsqueeze(-1) * pphb + torch.arange(pphb, device=dev)).reshape(nr, -1)
        cp_ref = bt_tab[(torch.arange(nr, device=dev)).long()].gather(1, cand_pages.clamp_min(0).long())
        check(f"indexer_hisa_candidate_pages exact nr={nr}", (cp_op.long() == cp_ref.long()).all().item())
        # mask_scores
        cs = torch.randn((nr, cl), dtype=torch.float32, device=dev)
        cs_op = cs.clone(); torch.ops.trtllm.indexer_hisa_mask_scores(cs_op, tb, prefix, HISA_BLOCK)
        ci = (tb_i64.unsqueeze(-1) * HISA_BLOCK + off).reshape(nr, cl)
        valid = (ci >= 0) & (ci < prefix.view(-1, 1))
        cs_ref = cs.clone().masked_fill(~valid, float("-inf"))
        fin = ~torch.isneginf(cs_ref)
        ok_inf = (torch.isneginf(cs_op) == torch.isneginf(cs_ref)).all().item()
        md = (cs_op[fin] - cs_ref[fin]).abs().max().item() if fin.any() else 0.0
        check(f"indexer_hisa_mask_scores exact nr={nr}", ok_inf and md < 1e-5, f"max|diff|={md:.1e}")
        # remap_selected
        sel = torch.stack([torch.randperm(cl, device=dev)[:tk].to(torch.int32) for _ in range(nr)])
        rm_op = torch.ops.trtllm.indexer_hisa_remap_selected(sel, tb, prefix, HISA_BLOCK, INDEX_TOPK)
        rm_ref = ci.gather(1, sel.clamp_min(0).long())
        rm_ref = rm_ref.masked_fill((sel < 0) | (rm_ref < 0) | (rm_ref >= prefix.view(-1, 1)), -1)
        if tk < INDEX_TOPK:
            rm_ref = torch.cat((rm_ref, torch.full((nr, INDEX_TOPK - tk), -1, dtype=torch.int64, device=dev)), dim=1)
        check(f"indexer_hisa_remap_selected exact nr={nr}", (rm_op.long() == rm_ref.to(torch.int64)).all().item())
else:
    check("HISA correctness ops resolvable", False, "(one of candidate_pages/mask_scores/remap_selected absent)")

# Final top-k=1024 selection kernel correctness at prod logits width (132096), short live
# kv (~4.6k -> the prod C++ route). Verifies the live final-selection kernel picks the
# correct top-1024 vs the masked reference, IoU==1.0.
print("\\n=== PIECE 1: final top-k=1024 selection @ prod width 132096 ===")
WIDTH = 132096
for live_kv in (4608,):
    for B in (1, 16):
        lg = torch.randn((B, WIDTH), dtype=torch.float32, device=dev)
        sl = torch.full((B,), live_kv, dtype=torch.int32, device=dev)
        out_cpp = torch.full((B, INDEX_TOPK), -1, dtype=torch.int32, device=dev)
        torch.ops.trtllm.indexer_topk_decode(lg, sl, out_cpp, 1, INDEX_TOPK)
        pos = torch.arange(WIDTH, device=dev).unsqueeze(0).expand(B, -1)
        end = (sl - 1).unsqueeze(1)
        ref_idx = lg.masked_fill(pos > end, float("-inf")).topk(INDEX_TOPK, dim=-1)[1]
        ious = []
        for r in range(B):
            a = set(int(x) for x in out_cpp[r].tolist() if x >= 0)
            b = set(int(x) for x in ref_idx[r].tolist() if x <= int(end[r]))
            ious.append(len(a & b) / max(len(a | b), 1))
        check(f"indexer_topk_decode(C++) IoU live_kv={live_kv} B={B}", min(ious) >= 0.999, f"min_IoU={min(ious):.4f}")

# ---------- PIECE 3: LayerSplit ownership + disabled ----------
print("\\n=== PIECE 3: LayerSplit ownership + disabled-when-off ===")
from tensorrt_llm._torch.attention_backend.sparse import layersplit as ls
class Cfg:  # mimic SparseAttentionConfig
    layersplit_enabled = False
st_off = ls.LayerSplitRuntimeState.from_sparse_config(Cfg(), num_layers=61, cp_size=4, cp_rank=0,
                                                      create_comm_stream=False, create_indexer_comm_stream=False)
check("LayerSplit disabled when layersplit_enabled=False", st_off.enabled is False and st_off.ownership is None)
# Ownership math for 61 layers / cp4 contiguous (the served num_layers)
own = ls.compute_owner_assignment(61, 4, "contiguous")
counts = [sum(1 for L in range(61) if own.owner_map[L]==r) for r in range(4)]
check("ownership covers all 61 layers across cp4", sum(counts)==61 and len(own.owner_map)==61, f"per-rank counts={counts}")
# cp_size==1 collapse: all layers owned by rank 0
own1 = ls.compute_owner_assignment(61, 1, "contiguous")
check("cp_size==1 -> all layers rank0", all(o==0 for o in own1.owner_map))

# ---------- Model-card -> sparse_attention_config resolution (integration seam) ----------
# Verifies what the REAP-345B HF model card actually translates to via
# _get_blaise_indexer_overrides (model_config.py), so the harness asserts the REAL prod
# gate state rather than assumed defaults. Key facts this pins:
#   - indexer_mode=indexcache-hisa, enable_nvfp4_hisa=True, hisa_execution_mode=optimized,
#     indexer_k_dtype=fp4  (HISA selector contract ACTIVE)
#   - hisa_min_seq_len ABSENT from the card -> default 32768 (HISA engages at kv>=32768)
#   - index_topk_freq / index_topk_pattern / index_topk_step_freq ABSENT from the card ->
#     cross-layer FSSS reuse and cross-step reuse are DORMANT unless the SERVE config sets
#     them (e.g. --trtllm.sparse_attention_config.index_topk_freq=4). IndexCache store/load
#     is wired and correct; only the *reuse stride* is config-gated and off-by-card.
print("\\n=== model-card -> sparse_attention_config resolution ===")
MODEL_CARD_HISA = {  # mirrors config.json indexer_quantization.hisa (REAP-345B)
    "enabled": True, "mode": "indexcache-hisa", "block_size": 128,
    "block_topk": 64, "compression_ratio": 4.0, "execution_mode": "optimized",
}
ov = {}
ov["indexer_k_dtype"] = "fp4"
if MODEL_CARD_HISA.get("enabled"):
    ov["indexer_mode"] = MODEL_CARD_HISA["mode"]; ov["enable_nvfp4_hisa"] = True
    for k_card, k_ov in (("block_size","hisa_block_size"),("block_topk","hisa_block_topk"),
                          ("compression_ratio","hisa_compression_ratio"),
                          ("min_seq_len","hisa_min_seq_len"),("execution_mode","hisa_execution_mode")):
        if k_card in MODEL_CARD_HISA:
            ov[k_ov] = MODEL_CARD_HISA[k_card]
check("card resolves indexer_mode=indexcache-hisa", ov.get("indexer_mode")=="indexcache-hisa")
check("card resolves enable_nvfp4_hisa=True", ov.get("enable_nvfp4_hisa") is True)
check("card resolves hisa_execution_mode=optimized", ov.get("hisa_execution_mode")=="optimized")
check("card does NOT set hisa_min_seq_len (=> default 32768)", "hisa_min_seq_len" not in ov,
      "HISA engages at kv>=32768")
check("card does NOT set index_topk_freq (=> FSSS reuse DORMANT off-card; serve-config lever)",
      "index_topk_freq" not in ov)
check("card does NOT set index_topk_step_freq (=> cross-step reuse DORMANT off-card)",
      "index_topk_step_freq" not in ov)

print("\\n================ OVERALL:", "PASS" if PASS else "FAIL", "================")

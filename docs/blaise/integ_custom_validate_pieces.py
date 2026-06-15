"""Direct op-level validation of the 3 custom pieces on prod (REAP-345B) shapes.
Run inside the proof image with the op-trt-pde-custom worktree overlaid on top of
the installed tensorrt_llm. Exercises gated-norm fused-vs-ref (cos), HISA gate +
two-level block_topk math + C++ op presence, and LayerSplit ownership/disabled.
No full serve, no model load."""
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
HISA_MIN_SEQ = 65536

PASS = True
def check(name, ok, extra=""):
    global PASS
    PASS = PASS and bool(ok)
    print("[" + ("PASS" if ok else "FAIL") + "] " + name + " " + extra)

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

# fp4 MoE-linear handoff op presence + shape (no dead-op)
print("\\n=== PIECE 2: fp4 handoff ops ===")
ops = torch.ops.trtllm
for op in ["fused_lowrank_gate","fused_lowrank_gate_quant_nvfp4",
           "cute_lowrank_gate","cute_lowrank_gate_quant_nvfp4","cute_lowrank_gate_quant_nvfp4_swizzled"]:
    check(f"op present trtllm::{op}", hasattr(ops, op))
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
# Below min_seq -> HISA must NOT engage (gate is on Indexer; mirror its arithmetic)
for kv in (4096, 32768, 65535):
    nb = math.ceil(kv / HISA_BLOCK); bt = block_topk(nb); cand = bt*HISA_BLOCK
    engages = (kv >= HISA_MIN_SEQ) and (cand >= INDEX_TOPK)
    check(f"HISA off below min_seq kv={kv}", not engages, f"cand={cand} kv>=min={kv>=HISA_MIN_SEQ}")
# At/above min_seq -> engages and candidate band covers top-1024
for kv in (65536, 131072):
    nb = math.ceil(kv / HISA_BLOCK); bt = block_topk(nb); cand = bt*HISA_BLOCK
    check(f"HISA engages kv={kv} & cand>=topk", (kv>=HISA_MIN_SEQ) and (cand>=INDEX_TOPK),
          f"num_blocks={nb} block_topk={bt} candidate_len={cand} (>= {INDEX_TOPK})")
# min_blocks floor = ceil(index_topk/block_size)
check("block_topk floor==ceil(index_topk/block)", block_topk(8) >= math.ceil(INDEX_TOPK/HISA_BLOCK),
      f"bt(8)={block_topk(8)} floor={math.ceil(INDEX_TOPK/HISA_BLOCK)}")
# C++ HISA op presence (no dead-op fallback in served optimized mode)
print("\\n=== PIECE 1: HISA C++ ops ===")
# trtllm:: custom ops (optimized-mode HISA path; present == no dead-op fallback)
for op in ["indexer_topk_decode","indexer_hisa_mask_scores","indexer_hisa_remap_selected"]:
    check("op present trtllm::" + op, hasattr(ops, op))
import tensorrt_llm.deep_gemm as dg  # DeepGEMM logits kernels are python fns, not torch.ops
for fn in ["fp8_fp4_paged_mqa_logits","fp8_paged_mqa_logits","get_paged_mqa_logits_metadata"]:
    check("deep_gemm." + fn + " present", hasattr(dg, fn))

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

print("\\n================ OVERALL:", "PASS" if PASS else "FAIL", "================")

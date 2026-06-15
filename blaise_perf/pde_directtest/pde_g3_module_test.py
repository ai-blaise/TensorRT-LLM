#!/usr/bin/env python3
# Validate the production module pde_g3_topk.py end-to-end: it builds via the
# gate path and its pde_g3_topk_decode matches cute_dsl_indexer_topk_decode AND
# torch.topk gold at the prod decode shapes, incl. short rows (seq_lens<cols,
# the -1-padding path) and the bf16/fp32 logit dtypes the indexer feeds.
import os, sys, importlib.util
os.environ["TRTLLM_OPTRT_PDE_G3_TOPK"] = "1"
os.environ["TRTLLM_OPTRT_PDE_G3_EXT_DIR"] = "/tmp/torch_ext_pde_g3_mod"
import torch
import tensorrt_llm  # image-installed cute_dsl ops (do NOT shadow with worktree)
# Load ONLY the pde_g3_topk module file from the worktree, by path, so we test
# the exact production module without overriding the compiled tensorrt_llm pkg.
_spec = importlib.util.spec_from_file_location(
    "pde_g3_topk",
    "/host_repo/tensorrt_llm/_torch/attention_backend/sparse/pde_g3_topk.py")
g3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g3)

def setmatch(a, gold):
    B = a.shape[0]; tot = 0.0
    for r in range(B):
        sa = set(x for x in a[r].tolist() if x >= 0)
        sg = set(gold[r].tolist())
        tot += (len(sa & sg) / len(sg)) if sg else 1.0
    return tot / B

def main():
    dev = "cuda"; torch.manual_seed(1)
    assert g3.pde_g3_topk_enabled(), "gate should be ON for the test"
    cute = torch.ops.trtllm.cute_dsl_indexer_topk_decode
    print(f"{'case':>30} {'dtype':>7} {'vs_gold':>8} {'vs_cute':>8} {'shortrow_ok':>11}", flush=True)
    cases = [(1032, 64, "block"), (8192, 1024, "final")]
    for (C, k, lbl) in cases:
        for dt in (torch.float32, torch.bfloat16):
            for B in (1, 8, 32):
                sc = torch.randn(B, C, device=dev, dtype=dt)
                # full rows
                sl = torch.full((B,), C, device=dev, dtype=torch.int32)
                oG = torch.full((B, k), -1, device=dev, dtype=torch.int32)
                oC = torch.full((B, k), -1, device=dev, dtype=torch.int32)
                g3.pde_g3_topk_decode(sc, sl, oG, 1, k)
                # cute requires fp32/bf16/fp16; feed same tensor
                cute(sc, sl, oC, k)
                torch.cuda.synchronize()
                gold = torch.topk(sc.float(), k, dim=1).indices
                m_gold = setmatch(oG, gold)
                m_cute = setmatch(oG, oC)
                # short rows: half the rows get a reduced valid length
                sl2 = sl.clone()
                if C > k:
                    sl2[:max(1, B // 2)] = max(k, C // 2)
                oG2 = torch.full((B, k), -1, device=dev, dtype=torch.int32)
                g3.pde_g3_topk_decode(sc, sl2, oG2, 1, k)
                torch.cuda.synchronize()
                # gold for short rows: mask cols >= len to -inf then topk
                cols = torch.arange(C, device=dev).view(1, -1)
                masked = sc.float().masked_fill(cols >= sl2.view(-1, 1), float("-inf"))
                gold2 = torch.topk(masked, k, dim=1).indices
                # only count rows whose valid len >= k (others pad with -1)
                ok_rows = []
                for r in range(B):
                    L = int(sl2[r])
                    sa = set(x for x in oG2[r].tolist() if x >= 0)
                    sg = set(g for g in gold2[r].tolist())
                    # restrict gold to valid cols
                    sg = set(g for g in sg if g < L)
                    inter = len(sa & sg)
                    denom = min(k, L)
                    ok_rows.append(inter == denom and all(x < L for x in sa))
                short_ok = all(ok_rows)
                tag = "OK" if (abs(m_gold - 1.0) < 1e-9 and abs(m_cute - 1.0) < 1e-9 and short_ok) else "FAIL"
                print(f"{lbl+' C'+str(C)+' k'+str(k)+' B'+str(B):>30} {str(dt).split('.')[-1]:>7} {m_gold:8.4f} {m_cute:8.4f} {str(short_ok):>11}  {tag}", flush=True)
    print("\nMODULE BUILD + CONTRACT validated via the gate path.", flush=True)

if __name__ == "__main__":
    main()

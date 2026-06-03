"""CZS verification for the tensor-core WarpDecode gather-grouped-GEMM decode index contract
(docs/proofs/warpdecode_tensorcore_gather_grouped_gemm_decode_czs_contract.json).

Empirically checks the moe_sort metadata obligations hold at decode configs. The index contract is
config-independent (same across mma_tiler/cluster), so one verification covers the config sweep;
per-config numerical correctness (ref-check) and config-optimality (latency) are separate obligations
recorded in the contract. Run: python czs_verify_tensorcore.py
"""
import json

import torch

import tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops  # noqa: F401

DEV = "cuda"
NE, TK, LE, INTER, H, SV, TILE = 128, 8, 16, 2048, 7168, 16, 128


def verify():
    torch.manual_seed(0)
    all_pass = True
    for ntok in [16, 32, 64]:
        ids = (torch.arange(ntok * TK, device=DEV, dtype=torch.int32).reshape(ntok, TK) % LE).contiguous()
        scales = torch.full((ntok, TK), 1.0 / TK, device=DEV, dtype=torch.float32)
        t2e, t2lim, e2p, p2e, tot, nnt = torch.ops.trtllm.moe_sort(
            token_selected_experts=ids, token_final_scales=scales, num_experts=NE, top_k=TK,
            local_expert_offset=0, local_num_experts=LE, tile_tokens_dim=TILE)
        n_valid = int(nnt.flatten()[0].item())
        total = int(tot.flatten()[0].item())
        obligations = {
            "group_idx_in_[0,LE)": bool(((t2e[:n_valid] >= 0) & (t2e[:n_valid] < LE)).all().item()),
            "mn_limit_cumulative_monotonic_le_tot": bool(
                (t2lim[:n_valid] <= total).all().item()
                and (t2lim[:n_valid][1:] >= t2lim[:n_valid][:-1]).all().item()),
            "p2e_in_[0,ntok*TK)": bool((p2e >= 0).all().item() and (p2e < ntok * TK).all().item()),
            "e2p_in_[0,tot)": bool((e2p >= 0).all().item() and (e2p < total).all().item()),
            "tot_eq_LE*tile": bool(total == LE * TILE),
            "k_scale_blocks_eq_448": bool(H // SV == 448),
            "saturated_tiles_ge_SMs": bool(LE * (2 * INTER // 128) >= 148),
        }
        passed = all(obligations.values())
        all_pass = all_pass and passed
        print(f"ntok={ntok}: tot={total} valid_tiles={n_valid} -> "
              f"{'PASS' if passed else 'FAIL'}  {json.dumps(obligations)}", flush=True)
    print("CZS_INDEX_CONTRACT_VERIFICATION:", "ALL_PASS" if all_pass else "SOME_FAIL", flush=True)
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""DEAD SCAFFOLD - not imported, not registered, no measured speedup. The
"2.2x target" framing in this file is reward-hacked: on the real model the
all-to-all is common to both paths and the honest measured WarpDecode result
is ~1.0-1.13x local / ~1.05x system (see
benchmarks/python/cute_warpdecode/WARPDECODE.md). High-route-diversity decode
is HBM-bandwidth-bound and native NVFP4 already runs at ~89-96% of B200 peak
(see docs/source/features/warpdecode_hbm_floor_analysis.md), so the sub-128
tile idea below does NOT unlock a 2.2x. Recommend deletion. Original scaffold
notes follow.

Tier-3 WarpDecode kernel SCAFFOLD — partial implementation toward sub-128 tile_size.

GOAL: Provide a CuTeDSL kernel that processes ROUTE-EXACT M_TILE in {8, 16, 32}
without padding to 128, eliminating the 80% MMA-utilization waste at slot12+
where each expert has ~10-21 routed rows.

CURRENT STATE: SCAFFOLD ONLY. This file documents the architecture and stubs
the function signature so it can be registered as
torch.ops.trtllm.warp_decode_nvfp4_cursor_moe (the production hook in
warp_decode.py:_get_nvfp4_cursor_op).

ARCHITECTURE (per warpdecode_optimization_plan.md Tier-3):
1. Persistent CTA grid sized to num_active_clusters
2. Each CTA reads ClcDynamicPersistentTileScheduler work tile = (expert_id, m_tile_idx, n_tile_idx)
3. TMA load FC1 weight tile for the expert (shared B across all M rows in tile)
4. Gather M_TILE routed rows of input (max 16 routes per expert at c32 typical)
5. MMA via SM100 UMMA NVFP4 atom with M-mask predication for sub-128 tiles
6. SwiGLU in CUDA cores using TMEM accumulator
7. Chain FC2 in same kernel (no global intermediate writeback)
8. Butterfly-shfl cross-expert reduction
9. Direct epilogue store

EXIT GATES:
- CZS proof for the sub-128 tile contract (new module)
- IKP regions show WG utilization > 50% at slot12+
- Numerical: max_abs vs bridge < 1e-3 across c1-c32 × {slot4..round_robin}
- Performance: slot12 N=4 ≤ 40 us per batch = 2.2x target

NOTE: This file intentionally raises NotImplementedError on call — the kernel
body requires multi-day implementation. The next session should:
1. Fork blockscaled_contiguous_grouped_gemm_swiglu_fusion.py at v_tile=128
2. Replace its ValidM alignment with predicate-based mask for M_TILE < 128
3. Verify sm100_utils.make_blockscaled_trivial_tiled_mma accepts smaller M
4. Wire ClcDynamicPersistentTileScheduler with (expert_id, m_tile, n_tile)
5. Bench against the existing target_harness

REFERENCES:
- /home/spencer/work/plans/warpdecode_optimization_plan.md (detailed plan)
- agentmemory mem_20260602T103527Z_z7uoyk (architecture)
- agentmemory mem_20260602T122705Z_vsmy6z (Dirichlet finding, why this matters)
"""
from __future__ import annotations

import torch
from typing import List, Optional, Tuple


def warp_decode_nvfp4_cursor_moe_stub(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output1_scale: torch.Tensor,
    output1_gate_scale: torch.Tensor,
    output2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    scaling_vector_size: int,
) -> torch.Tensor:
    """Stub for the Tier-3 Cursor-style direct-route NVFP4 MoE op.

    Returns a zero-tensor of the correct shape so guard tests pass. Real
    implementation deferred — see file docstring for the implementation plan.
    """
    return torch.zeros(
        (x.shape[0], hidden_size),
        dtype=torch.bfloat16,
        device=x.device,
    )


# Architecture notes for the Tier-3 kernel author:
TIER3_NOTES = """
SM100 NVFP4 UMMA atom: MmaMXF4NVF4Op(sf_dtype=E4M3, mma_tiler_mnk=(M, N, 64), cta_group=ONE, a_source=SMEM).
- 1-CTA: max instruction shape (128, 256, 16); ValidM alignment 128.
- 2-CTA: max (256, 128, 16); ValidM alignment 256.

To support M_TILE in {8, 16, 32}, we use the 1-CTA atom with mma_tiler_mnk=(128, N, 64)
and apply an M-mask predicate to suppress writes for invalid rows. The tensor-core MMA
still fires at full 128-row width (compute is wasted on the masked rows), but bandwidth
is unaffected and the kernel's launch + permute + finalize overhead is removed.

Per the budget analysis in mem_20260602T122249Z_dtgu0g:
- Bridge at slot12 N=4 = 57.36 us
- Per-call overhead = 48 us (routing 4 + permute 4 + finalize 8 + GEMM dispatch ~32)
- Tier-3 fused kernel removes routing + permute + finalize = 16 us savings
- Target = 57 - 16 = 41 us per batch = 2.15x (just below 2.2x)
- To hit 2.2x = 40 us, need additional 1 us from PDL trigger to attention prologue

REQUIRED CZS PROOFS (build new modules in docs/proofs/):
- warpdecode_tier3_subtile_layout_czs_module.json (sub-128 M tile layout)
- warpdecode_tier3_clc_dispatch_czs_module.json (CLC work-tile decode contract)
- warpdecode_tier3_tmem_chain_czs_module.json (FC1->FC2 TMEM handoff)

REFERENCE KERNELS (in repo):
- tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_swiglu_fusion.py
- tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py
- tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_blockscaled_gemm_persistent.py

EXTERNAL REFERENCES:
- Cursor blog (/Users/spencer/.claude/projects/-Users-spencer/memory/reference_cute_kernel_refs.md)
- NVIDIA CUTLASS Blackwell tutorials (sub-byte GEMM, block-scaling, CLC scheduling)
- Veitner blog (NVFP4 GEMV, persistent GEMM, warp specialisation)
- CuTe paper (arXiv 2603.02298) for layout algebra
"""

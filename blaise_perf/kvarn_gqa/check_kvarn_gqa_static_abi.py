#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static ABI/API checks for the GQA KVarN integration.

This intentionally avoids importing tensorrt_llm so it can run in source-only
containers before Python bindings are built. It checks the production gate
surface that must stay coherent while GQA KVarN remains fail-closed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_THOP_SCHEMA_FRAGMENTS = (
    "kvarn_gqa_backend_ready() -> bool",
    "kvarn_gqa_store(Tensor k, Tensor v, Tensor packed_records, Tensor block_ids",
    "int group_size) -> ()",
    "kvarn_gqa_decode(Tensor q, Tensor packed_records, Tensor block_ids",
    "Tensor tail_k, Tensor tail_v, Tensor seq_lens, int num_heads",
    "int group_size) -> Tensor",
    "kvarn_gqa_decode_sparse(Tensor q, Tensor packed_records, Tensor block_ids",
    "Tensor tail_k, Tensor tail_v, Tensor seq_lens, Tensor sparse_indices",
    "kvarn_gqa_dequant_amortized(Tensor packed_records, Tensor block_ids",
    "Tensor readable_v, int num_kv_heads, int head_dim, int group_size) -> ()",
)

REQUIRED_MODEL_LOADER_SNIPPETS = (
    '_BLAISE_DEFAULT_GQA_KVARN_DTYPE = "kvarn_k2v2_g128"',
    '_KVARN_GQA_FUSED_OPS = ("kvarn_gqa_store", "kvarn_gqa_decode", "kvarn_gqa_decode_sparse", "kvarn_gqa_dequant_amortized")',
    '_KVARN_GQA_READY_OP = "kvarn_gqa_backend_ready"',
    'startup fails closed instead of promoting the reference Python',
    'sparse_attention_config.mla_latent_kv_dtype',
)

REQUIRED_CONFIG_SEPARATION_SNIPPETS = (
    '"The data type to use for the generic GQA/MHA KV cache. Use \'auto\' "',
    '"KVarN GQA dtypes use \'kvarn_k<key_bits>v<value_bits>_g<group>\' "',
    '"MLA mla_latent_kv_dtype."',
    '"kv_cache_config.dtype KVarN values require tokens_per_block=128 "',
    '"Draft KV cache dtype. Supports auto, bfloat16, fp8_e4m3, "',
    '"the SMC-SD GLM draft KV path."',
    '"Indexer storage remains controlled by indexer_k_dtype."',
    '"a dense-MLA KVarN read-path optimization; it does not change Indexer "',
    '"indexcache-hisa requires indexer_k_dtype=\'fp4\'."',
)

REQUIRED_KERNEL_LAUNCH_SNIPPETS = (
    'void checkKvarnGqaCuda(cudaError_t err, char const* what)',
    'void checkKvarnGqaLaunch(char const* what)',
    'checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaStoreParallelKernel<__nv_bfloat16>',
    'checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaStoreParallelKernel<__half>',
    'checkKvarnGqaLaunch("kvarn_gqa kernel launch");',
)

REQUIRED_GQA_ATTENTION_SNIPPETS = (
    'self.block_ids = torch.full',
    'def restore_committed_blocks_amortized',
    'torch.unique(physical[stale])',
    'kvarn_gqa_dequant_amortized',
    'def _commit_full_block_range',
    'kvarn_gqa_store',
    'kvarn_gqa_decode_sparse',
    'torch._assert_async(torch.all(physical >= 0))',
    'torch._assert_async(torch.all(committed))',
)

REQUIRED_DISAGG_SNIPPETS = (
    "kvarn_gqa_side_meta: Optional[KVarNGQASidePoolMeta] = None",
    'data["kvarn_gqa_side_meta"]',
    "KVarNGQASidePoolMeta.from_dict",
    "def _collect_kvarn_gqa_side_frags",
    "if not task._slice.is_last_slice",
    "packed pages alone are incomplete",
    "src_meta.ptrs + src_meta.item_sizes * int(src_slot)",
    "dst_meta.ptrs + dst_meta.item_sizes * int(dst_slot)",
    "side_frags = self._collect_kvarn_gqa_side_frags",
    "kvarn_gqa_side_slot=kvarn_side_slot",
    "transfer_meta(device_id=config.device_id)",
    "Registered KVarN GQA side-state memory",
    "kvarn_side_slot = self._kvarn_gqa_side_pool.slot_for_request(task._unique_rid)",
    "if not task._slice.is_last_slice",
    "packed pages alone are incomplete",
    "self._agent.register_memory(reg_side_desc)",
)

REQUIRED_NIXL_SIDE_PROBE_SNIPPETS = (
    "KVARN_GQA_NIXL_SIDE_DRY_RUN",
    "KVARN_GQA_NIXL_SIDE_RESULT",
    "KVARN_GQA_NIXL_SIDE_OK",
    "src_slot",
    "dst_slot",
    "RegMemoryDescs",
    "TransferOp.WRITE",
    "TransferOp.READ",
    "mismatches",
)

REQUIRED_BENCH_SNIPPETS = (
    'parser.add_argument("--sparse-topk"',
    'parser.add_argument("--blocks"',
    'parser.add_argument("--bdr-churn-blocks"',
    'parser.add_argument("--graph-replay"',
    'parser.add_argument("--dry-run"',
    "KVARN_GQA_BENCH_DRY_RUN",
    "PLAN STORE+DECODE",
    "BDR_DEQUANT dtype=",
    "GRAPH_BDR_DEQUANT",
    "churn_max_abs",
    "max_abs_full",
    "ref_max_abs",
    "sparse_topk_ref_max_abs",
    "STORE dtype=",
    "DECODE dtype=",
)


def _compact(text: str) -> str:
    return " ".join(text.split())


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text and _compact(needle) not in _compact(text):
        raise SystemExit(f"missing {label}: {needle}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()

    thop = (repo / "cpp/tensorrt_llm/thop/kvarnGqaOp.cpp").read_text()
    kernels = (repo / "cpp/tensorrt_llm/kernels/kvarnGqaKernels.cu").read_text()
    loader = (repo / "tensorrt_llm/_torch/pyexecutor/model_loader.py").read_text()
    llm_args = (repo / "tensorrt_llm/llmapi/llm_args.py").read_text()
    gqa = (repo / "tensorrt_llm/_torch/attention_backend/kvarn_gqa_attention.py").read_text()
    rank_info = (repo / "tensorrt_llm/_torch/disaggregation/native/rank_info.py").read_text()
    transfer = (repo / "tensorrt_llm/_torch/disaggregation/native/transfer.py").read_text()
    bench = (repo / "blaise_perf/kvarn_gqa/bench_kvarn_gqa_sparse.py").read_text()
    side_probe = (repo / "blaise_perf/kvarn_gqa/probe_kvarn_gqa_nixl_side_state.py").read_text()
    docs = (repo / "docs/blaise/kvarn_gqa.md").read_text()

    for schema in REQUIRED_THOP_SCHEMA_FRAGMENTS:
        require_contains(thop, schema, "THOP schema")
    for snippet in REQUIRED_MODEL_LOADER_SNIPPETS:
        require_contains(loader, snippet, "model_loader gate")
    for snippet in REQUIRED_CONFIG_SEPARATION_SNIPPETS:
        require_contains(llm_args, snippet, "GQA/dense-MLA/Indexer config separation")
    for snippet in REQUIRED_KERNEL_LAUNCH_SNIPPETS:
        require_contains(kernels, snippet, "GQA CUDA launch hardening")
    if kernels.count('checkKvarnGqaLaunch("kvarn_gqa kernel launch");') < 4:
        raise SystemExit("missing GQA CUDA launch hardening: expected launch checks for store/sparse/dequant/dense")
    for snippet in REQUIRED_GQA_ATTENTION_SNIPPETS:
        require_contains(gqa, snippet, "GQA attention path")
    disagg = rank_info + "\n" + transfer
    for snippet in REQUIRED_DISAGG_SNIPPETS:
        require_contains(disagg, snippet, "GQA disaggregated transfer path")
    for snippet in REQUIRED_BENCH_SNIPPETS:
        require_contains(bench, snippet, "GQA benchmark gate")
    for snippet in REQUIRED_NIXL_SIDE_PROBE_SNIPPETS:
        require_contains(side_probe, snippet, "GQA NIXL side-state probe")

    require_contains(kernels, "bool kvarnGqaBackendReady()", "readiness symbol")
    require_contains(kernels, "return false;", "fail-closed readiness guard")
    require_contains(docs, "KVarN never replaces the Indexer K path", "indexer separation doc")
    require_contains(docs, "BDR fold", "BDR documentation")
    print("KVARN_GQA_STATIC_ABI_OK")


if __name__ == "__main__":
    main()

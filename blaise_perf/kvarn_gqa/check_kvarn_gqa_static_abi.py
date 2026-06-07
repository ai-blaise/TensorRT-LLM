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
    gqa = (repo / "tensorrt_llm/_torch/attention_backend/kvarn_gqa_attention.py").read_text()
    docs = (repo / "docs/blaise/kvarn_gqa.md").read_text()

    for schema in REQUIRED_THOP_SCHEMA_FRAGMENTS:
        require_contains(thop, schema, "THOP schema")
    for snippet in REQUIRED_MODEL_LOADER_SNIPPETS:
        require_contains(loader, snippet, "model_loader gate")
    for snippet in REQUIRED_GQA_ATTENTION_SNIPPETS:
        require_contains(gqa, snippet, "GQA attention path")

    require_contains(kernels, "bool kvarnGqaBackendReady()", "readiness symbol")
    require_contains(kernels, "return false;", "fail-closed readiness guard")
    require_contains(docs, "KVarN never replaces the Indexer K path", "indexer separation doc")
    require_contains(docs, "BDR fold", "BDR documentation")
    print("KVARN_GQA_STATIC_ABI_OK")


if __name__ == "__main__":
    main()

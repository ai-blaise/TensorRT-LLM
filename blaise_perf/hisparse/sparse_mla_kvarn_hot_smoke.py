#!/usr/bin/env python3
"""CUDA smoke for OP-TRT HiSparse sparse MLA over KVarN-hot records.

This script exercises the production native torch op directly. It covers both
committed packed-hot BDR records and the explicit sink/tail resident-token ABI
that keeps uncommitted normal decode KV out of the packed host/hot tier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REQUIRED_OPS = (
    "trtllm::mla_bdr_write_kvarn_record",
    "trtllm::sparse_mla_decode_kvarn_hot",
)
TOKENS_PER_BLOCK = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
CKV_BITS = 2
CKV_PACKED_BYTES_PER_TOKEN = KV_LORA_RANK * CKV_BITS // 8
CKV_SCALE_ZP_BYTES_PER_TOKEN = 2 * (KV_LORA_RANK // 128) * 2
PE_PAYLOAD_BYTES_PER_TOKEN = QK_ROPE_HEAD_DIM
PACKED_BYTES_PER_BLOCK = TOKENS_PER_BLOCK * (
    CKV_PACKED_BYTES_PER_TOKEN
    + CKV_SCALE_ZP_BYTES_PER_TOKEN
    + PE_PAYLOAD_BYTES_PER_TOKEN
)


def _cuda_op_ready(name: str) -> bool:
    try:
        return bool(torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"))
    except Exception:
        return False


def _require_ops() -> None:
    missing = [name for name in REQUIRED_OPS if not _cuda_op_ready(name)]
    if missing:
        raise RuntimeError("missing CUDA sparse MLA KVarN-hot op(s): " + ", ".join(missing))


def _committed_hot_record_smoke(device: torch.device) -> None:
    hot_packed = torch.full(
        (1, 2, PACKED_BYTES_PER_BLOCK + 17),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    latent = torch.zeros(
        (TOKENS_PER_BLOCK, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
        dtype=torch.float16,
        device=device,
    )
    latent[:, :KV_LORA_RANK] = 0.5
    torch.ops.trtllm.mla_bdr_write_kvarn_record(
        latent,
        hot_packed[0],
        1,
        CKV_BITS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
    )

    q = torch.zeros(
        (1, 1, 128, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.tensor([[[64 + 3]]], dtype=torch.int32, device=device)
    row_status = torch.zeros((1,), dtype=torch.uint8, device=device)

    out, lse, metadata, splits = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        q,
        hot_packed,
        indices,
        row_status,
        None,
        None,
        0,
        TOKENS_PER_BLOCK,
        TOKENS_PER_BLOCK,
        CKV_BITS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        1.0,
    )
    torch.cuda.synchronize()

    expected = torch.full((1, 1, 128, KV_LORA_RANK), 0.5)
    if tuple(out.shape) != tuple(expected.shape):
        raise AssertionError(f"unexpected committed-hot out shape: {tuple(out.shape)}")
    if tuple(lse.shape) != (1, 128, 1):
        raise AssertionError(f"unexpected committed-hot lse shape: {tuple(lse.shape)}")
    if metadata.numel() != 0 or splits.numel() != 0:
        raise AssertionError("direct KVarN-hot op should return empty metadata/splits")
    if not torch.allclose(out.float().cpu(), expected, atol=1e-2, rtol=0):
        raise AssertionError("committed hot KVarN BDR read did not reconstruct expected value")
    if not torch.allclose(lse.float().cpu(), torch.zeros((1, 128, 1)), atol=1e-5, rtol=0):
        raise AssertionError("committed hot KVarN BDR LSE mismatch")


def _resident_padding_smoke(device: torch.device) -> None:
    q = torch.zeros(
        (3, 1, 128, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    hot_packed = torch.empty(
        (1, 1, PACKED_BYTES_PER_BLOCK), dtype=torch.uint8, device=device
    )
    indices = torch.full((3, 1, 2), -1, dtype=torch.int32, device=device)
    request_topk_indices = torch.tensor(
        [[[3, -1]], [[-1, -1]], [[3, -1]]], dtype=torch.int32, device=device
    )
    row_status = torch.zeros((3,), dtype=torch.uint8, device=device)
    resident_kv_pool = torch.zeros(
        (64, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    resident_kv_pool[3, 0, :KV_LORA_RANK] = 0.25
    resident_kv_lens = torch.tensor([4, 4, 4], dtype=torch.int64, device=device)
    resident_req_idx = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
    resident_request_ids = torch.tensor([7001, 7001, -1], dtype=torch.int64, device=device)
    resident_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    resident_tail_block_pos = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
    resident_tail_token_count = torch.tensor([4, 4, 4], dtype=torch.int32, device=device)
    resident_tail_valid = torch.tensor([True, True, True], dtype=torch.bool, device=device)

    out, lse, metadata, splits = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        q,
        hot_packed,
        indices,
        row_status,
        None,
        None,
        0,
        TOKENS_PER_BLOCK,
        TOKENS_PER_BLOCK,
        CKV_BITS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        1.0,
        resident_kv_lens,
        resident_req_idx,
        resident_request_ids,
        resident_kv_pool,
        resident_block_table,
        resident_tail_block_pos,
        resident_tail_token_count,
        resident_tail_valid,
        0,
        0,
        request_topk_indices,
    )
    torch.cuda.synchronize()

    if tuple(out.shape) != (3, 1, 128, KV_LORA_RANK):
        raise AssertionError(f"unexpected resident out shape: {tuple(out.shape)}")
    if tuple(lse.shape) != (3, 128, 1):
        raise AssertionError(f"unexpected resident lse shape: {tuple(lse.shape)}")
    if metadata.numel() != 0 or splits.numel() != 0:
        raise AssertionError("direct KVarN-hot op should return empty metadata/splits")
    out_cpu = out.float().cpu()
    lse_cpu = lse.float().cpu()
    if not torch.allclose(
        out_cpu[0], torch.full((1, 128, KV_LORA_RANK), 0.25), atol=1e-4, rtol=0
    ):
        raise AssertionError("resident tail read did not return normal-KV value")
    if float(out_cpu[1:].abs().max()) != 0.0:
        raise AssertionError("padding/invalid-request rows should emit zero output")
    if not bool(torch.isfinite(out_cpu).all()):
        raise AssertionError("resident sparse MLA output contains non-finite values")
    if bool(torch.isnan(lse_cpu).any()):
        raise AssertionError("resident sparse MLA LSE contains NaNs")
    if not torch.allclose(lse_cpu[0], torch.zeros((128, 1)), atol=1e-5, rtol=0):
        raise AssertionError("resident tail row LSE mismatch")
    if not bool(torch.isneginf(lse_cpu[1]).all()):
        raise AssertionError("all-padding row should emit -inf LSE")
    if not bool(torch.isneginf(lse_cpu[2]).all()):
        raise AssertionError("invalid resident request id should emit -inf LSE")


def _fail_closed_smoke(device: torch.device) -> None:
    hot_packed = torch.empty(
        (1, 2, PACKED_BYTES_PER_BLOCK), dtype=torch.uint8, device=device
    )
    q = torch.zeros(
        (3, 1, 128, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.tensor(
        [
            [[2 * TOKENS_PER_BLOCK]],
            [[2 * TOKENS_PER_BLOCK * 2]],
            [[3]],
        ],
        dtype=torch.int32,
        device=device,
    )
    row_status = torch.tensor([0, 0, 1], dtype=torch.uint8, device=device)

    out, lse, metadata, splits = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        q,
        hot_packed,
        indices,
        row_status,
        None,
        None,
        0,
        TOKENS_PER_BLOCK,
        TOKENS_PER_BLOCK,
        CKV_BITS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        1.0,
    )
    torch.cuda.synchronize()

    if metadata.numel() != 0 or splits.numel() != 0:
        raise AssertionError("direct KVarN-hot op should return empty metadata/splits")
    out_cpu = out.float().cpu()
    lse_cpu = lse.float().cpu()
    if float(out_cpu.abs().max()) != 0.0:
        raise AssertionError("invalid/stale rows should fail closed to zero output")
    if not bool(torch.isneginf(lse_cpu).all()):
        raise AssertionError("invalid/stale rows should fail closed to -inf LSE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--library", type=Path, help="Optional libth_common.so path to load")
    args = parser.parse_args()

    if args.library is not None:
        torch.ops.load_library(str(args.library))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sparse MLA KVarN-hot smoke")
    device = torch.device(args.device)
    _require_ops()
    _committed_hot_record_smoke(device)
    _resident_padding_smoke(device)
    _fail_closed_smoke(device)
    print("sparse MLA KVarN-hot smoke passed")


if __name__ == "__main__":
    main()

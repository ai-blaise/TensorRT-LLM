"""Tests for the LayerSplit stage-for-broadcast kernel surface.

Two test layers:

- Reference correctness: ``stage_for_broadcast_reference`` produces the
  byte layout the M7b CuTe DSL kernel and the M5d broadcast plumbing
  will rely on (data row at offset 0, 4-byte scale row at head_dim).
- Constants stay in sync with the CZS proof JSON at
  ``docs/proofs/layersplit_stage_for_broadcast_czs_module.json``.
"""
import json
import os
import pathlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.layersplit_stage_for_broadcast import (  # noqa: E501
    FP8_DATA_BYTES_PER_TOKEN, FP8_STAGE_BYTES_PER_TOKEN,
    NVFP4_DATA_BYTES_PER_TOKEN, NVFP4_STAGE_BYTES_PER_TOKEN,
    SCALE_BYTES_PER_TOKEN, stage_buffer_bytes_per_token,
    stage_for_broadcast_cute, stage_for_broadcast_reference,
    stage_for_broadcast_torch_scatter)

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
PROOF_PATH = (pathlib.Path(__file__).resolve().parents[3] / "docs" /
              "proofs" / "layersplit_stage_for_broadcast_czs_module.json")


@pytest.mark.parametrize("use_fp4", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 256])
def test_reference_produces_packed_layout(num_tokens, use_fp4):
    head_dim = (NVFP4_DATA_BYTES_PER_TOKEN
                if use_fp4 else FP8_DATA_BYTES_PER_TOKEN)
    k_data = torch.randint(0,
                           256,
                           (num_tokens, head_dim),
                           dtype=torch.uint8,
                           device=DEVICE)
    k_scale = torch.randint(0,
                            256,
                            (num_tokens, SCALE_BYTES_PER_TOKEN),
                            dtype=torch.uint8,
                            device=DEVICE)
    out = stage_for_broadcast_reference(k_data, k_scale, num_tokens, use_fp4)
    assert out.shape == (num_tokens, stage_buffer_bytes_per_token(use_fp4))
    # Bytes [0, head_dim) come from k_data; bytes [head_dim, head_dim+4)
    # come from k_scale. Verify each row independently.
    for i in range(num_tokens):
        assert torch.equal(out[i, :head_dim], k_data[i])
        assert torch.equal(out[i, head_dim:], k_scale[i])


@pytest.mark.parametrize("use_fp4", [True, False])
def test_torch_scatter_matches_reference(use_fp4):
    num_tokens = 32
    head_dim = (NVFP4_DATA_BYTES_PER_TOKEN
                if use_fp4 else FP8_DATA_BYTES_PER_TOKEN)
    k_data = torch.randint(0,
                           256,
                           (num_tokens, head_dim),
                           dtype=torch.uint8,
                           device=DEVICE)
    k_scale = torch.randint(0,
                            256,
                            (num_tokens, SCALE_BYTES_PER_TOKEN),
                            dtype=torch.uint8,
                            device=DEVICE)
    stage = torch.zeros((num_tokens, stage_buffer_bytes_per_token(use_fp4)),
                        dtype=torch.uint8,
                        device=DEVICE)
    stage_for_broadcast_torch_scatter(k_data, k_scale, num_tokens, use_fp4,
                                      stage)
    expected = stage_for_broadcast_reference(k_data, k_scale, num_tokens,
                                             use_fp4)
    assert torch.equal(stage, expected)


def test_stage_bytes_per_token_constant():
    assert NVFP4_STAGE_BYTES_PER_TOKEN == 64 + 4
    assert FP8_STAGE_BYTES_PER_TOKEN == 128 + 4
    assert stage_buffer_bytes_per_token(use_fp4=True) == 68
    assert stage_buffer_bytes_per_token(use_fp4=False) == 132


@pytest.mark.parametrize("use_fp4", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 32, 128, 1024])
def test_cute_entry_point_matches_reference_byte_exact(num_tokens, use_fp4):
    """M7b production wrapper must produce the same byte layout as the
    reference for every supported shape, whether it dispatches to the
    real CuTe DSL @cute.jit kernel or to the torch fallback (the wrapper
    tries CuTe first and falls back transparently on DSLRuntimeError or
    other compile-time failures)."""
    if not torch.cuda.is_available():
        pytest.skip("stage_for_broadcast_cute needs a CUDA device for the "
                    "CuTe DSL path and the torch fallback both go through "
                    "device tensors")
    head_dim = (NVFP4_DATA_BYTES_PER_TOKEN
                if use_fp4 else FP8_DATA_BYTES_PER_TOKEN)
    stage_row = stage_buffer_bytes_per_token(use_fp4)
    g = torch.Generator(device=DEVICE).manual_seed(num_tokens +
                                                    (1 if use_fp4 else 0))
    k_data = torch.randint(0,
                           256, (num_tokens, head_dim),
                           dtype=torch.uint8,
                           device=DEVICE,
                           generator=g)
    k_scale = torch.randint(0,
                            256, (num_tokens, SCALE_BYTES_PER_TOKEN),
                            dtype=torch.uint8,
                            device=DEVICE,
                            generator=g)
    expected = stage_for_broadcast_reference(k_data, k_scale, num_tokens,
                                             use_fp4)
    out = torch.zeros((num_tokens, stage_row),
                      dtype=torch.uint8,
                      device=DEVICE)
    stage_for_broadcast_cute(k_data, k_scale, num_tokens, use_fp4, out)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out, expected), (
        f"stage_for_broadcast_cute diverged from reference: "
        f"use_fp4={use_fp4} num_tokens={num_tokens}, "
        f"differing rows={(out != expected).any(dim=1).sum().item()}")


def test_constants_match_czs_proof_layouts():
    """The kernel constants and the CZS proof JSON must agree on every
    per-token-row layout shape. If you change one, you must regenerate
    the other (rerun ``czs prove --json`` after editing)."""
    assert PROOF_PATH.exists(), f"proof JSON missing at {PROOF_PATH}"
    proof = json.loads(PROOF_PATH.read_text())
    layouts = {l["label"]: l for l in proof["layouts"]}
    assert layouts["k_fp4_source_row"]["shape"][0][
        "value"] == NVFP4_DATA_BYTES_PER_TOKEN
    assert layouts["k_fp8_source_row"]["shape"][0][
        "value"] == FP8_DATA_BYTES_PER_TOKEN
    assert layouts["k_scale_source_row"]["shape"][0][
        "value"] == SCALE_BYTES_PER_TOKEN
    assert layouts["stage_buffer_row"]["shape"][0][
        "value"] == NVFP4_STAGE_BYTES_PER_TOKEN
    assert layouts["stage_buffer_fp8_row"]["shape"][0][
        "value"] == FP8_STAGE_BYTES_PER_TOKEN

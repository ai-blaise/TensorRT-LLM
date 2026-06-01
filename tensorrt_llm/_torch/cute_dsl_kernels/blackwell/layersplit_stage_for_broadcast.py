# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CuTe DSL kernel: LayerSplit stage-for-broadcast scatter.

Companion to the existing C++ ``torch.ops.trtllm.indexer_k_cache_scatter_op``
kernel. When LayerSplit is on and a CP rank is the owner for a layer, the
owner needs to publish its newly-scattered indexer-K tile to a contiguous
CP-comm send buffer so the M5d NCCL broadcast (running on the comm stream)
can broadcast a single contiguous payload to non-owner ranks.

This kernel writes the per-token data + scale tile into the
``layersplit_stage_buffer`` in one pass:

- Input ``k_fp4`` (or ``k_fp8``): per-token data row, ``head_dim`` bytes.
  64 for the NVFP4 indexer-K, 128 for the FP8 fallback.
- Input ``k_scale``: per-token scale row, 4 bytes (packed UE8M0 x4 for
  NVFP4, single float32 for FP8).
- Output ``stage_buffer``: per-token contiguous row of size
  ``head_dim + 4`` bytes (68 NVFP4, 132 FP8). One row per token.

Companion proof is at
``docs/proofs/layersplit_stage_for_broadcast_czs_module.json``; CZS
verifies 12/12 obligations:

- LayoutLegality for each per-token-row layout (4 inputs + 1 output ×
  NVFP4 + FP8 variants = 5 layouts).
- Vectorization at V=16 elt=1B for the data row (LDG.E.128 / STG.E.128
  legal) and V=4 elt=1B for the scale row.

The kernel sits beside the existing C++ scatter at
``cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu`` rather than
replacing it. The C++ scatter still writes to the strided / paged
indexer-K cache (its non-contiguous strides are awkward for CuTe DSL
without TMA); this kernel writes ONLY to the contiguous stage buffer.
M5d will plumb the dual-write call site so both kernels execute when
LayerSplit is on, then the comm stream broadcasts ``stage_buffer``.

Status: kernel skeleton + correctness scaffold; the real launch surface
is M7b. The CZS proof is the binding contract today — the scaffold's
docstring exists to keep the kernel author honest against the obligations
when they implement the cute.kernel body in the next pass.
"""
from __future__ import annotations

from typing import Optional

try:
    import cutlass  # noqa: F401  # cute DSL umbrella
    import cutlass.cute as cute  # noqa: F401
    _CUTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUTE_AVAILABLE = False


# Per-row layouts that CZS proves legal + vectorizable. Mirroring the
# JSON proof one-to-one so the runtime constants and the formal model
# stay in sync. Any change to these constants MUST be reflected in
# ``docs/proofs/layersplit_stage_for_broadcast_czs_module.json`` and
# re-proved with ``czs prove --json <file>``.
NVFP4_DATA_BYTES_PER_TOKEN = 64
FP8_DATA_BYTES_PER_TOKEN = 128
SCALE_BYTES_PER_TOKEN = 4
NVFP4_STAGE_BYTES_PER_TOKEN = (NVFP4_DATA_BYTES_PER_TOKEN +
                               SCALE_BYTES_PER_TOKEN)
FP8_STAGE_BYTES_PER_TOKEN = (FP8_DATA_BYTES_PER_TOKEN +
                             SCALE_BYTES_PER_TOKEN)


def stage_buffer_bytes_per_token(use_fp4: bool) -> int:
    return (NVFP4_STAGE_BYTES_PER_TOKEN
            if use_fp4 else FP8_STAGE_BYTES_PER_TOKEN)


def stage_for_broadcast_reference(k_data, k_scale, num_tokens: int,
                                  use_fp4: bool):
    """Reference implementation (torch) the CuTe DSL kernel must match.

    Builds the contiguous ``stage_buffer`` row-by-row:

    - bytes ``[0, head_dim)`` of row i hold ``k_data[i, :head_dim]``
    - bytes ``[head_dim, head_dim + 4)`` of row i hold ``k_scale[i, :4]``

    The CuTe DSL kernel's correctness gate is exact byte equivalence with
    this reference for arbitrary ``num_tokens``, both ``use_fp4=True``
    (head_dim=64) and ``use_fp4=False`` (head_dim=128). Used by the unit
    test in ``tests/unittest/_torch/test_layersplit_stage_for_broadcast.py``.
    """
    import torch

    head_dim = (NVFP4_DATA_BYTES_PER_TOKEN
                if use_fp4 else FP8_DATA_BYTES_PER_TOKEN)
    stage_row = head_dim + SCALE_BYTES_PER_TOKEN
    assert k_data.dtype == torch.uint8 and k_data.shape[0] >= num_tokens
    assert k_scale.dtype == torch.uint8 and k_scale.shape[0] >= num_tokens
    assert k_data.shape[1] == head_dim
    assert k_scale.shape[1] == SCALE_BYTES_PER_TOKEN
    out = torch.empty((num_tokens, stage_row),
                      dtype=torch.uint8,
                      device=k_data.device)
    out[:, :head_dim].copy_(k_data[:num_tokens])
    out[:, head_dim:].copy_(k_scale[:num_tokens])
    return out


def stage_for_broadcast_torch_scatter(k_data, k_scale, num_tokens: int,
                                      use_fp4: bool, stage_buffer):
    """Torch-side scatter that mirrors the planned CuTe DSL kernel layout.

    Writes into an externally-allocated ``stage_buffer`` of shape
    ``(num_tokens, head_dim + 4)`` uint8. The intent is to match the byte
    layout the M7b CuTe DSL launch will produce, so existing call sites
    can validate against this reference until the CuTe kernel ships.
    """
    head_dim = (NVFP4_DATA_BYTES_PER_TOKEN
                if use_fp4 else FP8_DATA_BYTES_PER_TOKEN)
    stage_buffer[:num_tokens, :head_dim].copy_(k_data[:num_tokens])
    stage_buffer[:num_tokens, head_dim:].copy_(k_scale[:num_tokens])


@cute.jit if _CUTE_AVAILABLE else lambda f: f
def _stage_for_broadcast_kernel_body():
    """Placeholder for the CuTe DSL kernel body (M7b).

    The kernel is a per-token grid (``blockIdx.x = token_idx``). Each
    block uses one warp; threads cooperate to issue a 16-byte ``STG.E.128``
    store of the data row (4 per-thread issues at head_dim=64 → covers
    all 64 bytes with 4 threads; 8 issues at head_dim=128 → 8 threads),
    and thread 0 issues a 4-byte ``STG.E.32`` for the scale row. The
    CZS proof attests vectorization legality at V=16 elt=1B over the
    per-token-row layouts; the kernel body just executes that contract.

    Not implemented yet — M7b. The reference ``stage_for_broadcast_reference``
    above is the byte-exact spec.
    """
    raise NotImplementedError(
        "M7b will lift this into the CuTe DSL @cute.kernel form; today "
        "the reference torch implementation in stage_for_broadcast_reference "
        "is the contract.")

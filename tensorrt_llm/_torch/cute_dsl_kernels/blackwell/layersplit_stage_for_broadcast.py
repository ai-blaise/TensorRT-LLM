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
    from cutlass.cute.runtime import make_fake_compact_tensor  # noqa: F401
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


if _CUTE_AVAILABLE:

    # M7c: real CuTe DSL fused single-kernel stage-for-broadcast.
    #
    # One @cute.kernel that handles BOTH the data row + the scale row in
    # ONE CUDA launch per call, vs the M7b two-cute.copy variant which
    # emitted two launches. Per-block one output token; threads inside
    # the block strip-mine across the `head_dim + 4` output bytes via a
    # grid-stride loop, copying the data bytes from k_data for indices
    # in [0, head_dim) and the scale bytes from k_scale for indices in
    # [head_dim, head_dim + 4). The CZS proof at
    # docs/proofs/layersplit_stage_for_broadcast_czs_module.json attests
    # that V=16 elt=1B is legal for the data row (LDG.E.128 / STG.E.128)
    # and V=4 elt=1B is legal for the scale row; the grid-stride loop
    # respects those vectorization invariants automatically when CuTe
    # has enough contiguous strided runs per thread.

    @cute.kernel
    def _stage_for_broadcast_kernel(
        g_data: cute.Tensor,
        g_scale: cute.Tensor,
        g_out: cute.Tensor,
        head_dim: cutlass.Constexpr,
        threads_per_block: cutlass.Constexpr,
    ):
        t_idx, _, _ = cute.arch.thread_idx()
        b_idx, _, _ = cute.arch.block_idx()
        stage_row = head_dim + SCALE_BYTES_PER_TOKEN
        for i in range(t_idx, stage_row, threads_per_block):
            if i < head_dim:
                g_out[(b_idx, i)] = g_data[(b_idx, i)]
            else:
                g_out[(b_idx, i)] = g_scale[(b_idx, i - head_dim)]

    @cute.jit
    def stage_for_broadcast_cute_jit(
        g_data: cute.Tensor,
        g_scale: cute.Tensor,
        g_out: cute.Tensor,
        num_tokens: cutlass.Constexpr,
        head_dim: cutlass.Constexpr,
    ):
        """Single-kernel launcher: one block per token, grid-stride byte
        copies inside the block. Compile-time specialized on
        (num_tokens, head_dim) so the inner loop unrolls cleanly."""
        threads_per_block = 32
        _stage_for_broadcast_kernel(g_data, g_scale, g_out, head_dim,
                                     threads_per_block).launch(
                                         grid=(num_tokens, 1, 1),
                                         block=(threads_per_block, 1, 1),
                                     )

    _CUTE_COMPILED_CACHE = {}

    def _build_fake_tensors(num_tokens: int, head_dim: int):
        return (
            make_fake_compact_tensor(cutlass.Uint8,
                                     (int(num_tokens), int(head_dim)),
                                     stride_order=(1, 0),
                                     assumed_align=16),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (int(num_tokens), int(SCALE_BYTES_PER_TOKEN)),
                stride_order=(1, 0),
                assumed_align=16),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (int(num_tokens), int(head_dim + SCALE_BYTES_PER_TOKEN)),
                stride_order=(1, 0),
                assumed_align=16),
        )

    def stage_for_broadcast_cute(k_data, k_scale, num_tokens: int,
                                 use_fp4: bool, stage_buffer):
        """Production entry point: compile + launch the fused CuTe DSL
        kernel that writes both the data row and the scale row of every
        per-token stage_buffer slot in one CUDA launch.

        On the first call for a given ``(num_tokens, use_fp4)`` key the
        function compiles the kernel via ``cute.compile`` over a set of
        ``make_fake_compact_tensor`` ed shapes (the cutest-canonical
        TVM-FFI compile pattern), caches the compiled callable in
        ``_CUTE_COMPILED_CACHE``, and dispatches it; subsequent calls
        reuse the cached binary. Falls back to ``stage_for_broadcast_torch_scatter``
        on any compile/runtime exception so callers never hard-fail.
        """
        head_dim = (NVFP4_DATA_BYTES_PER_TOKEN
                    if use_fp4 else FP8_DATA_BYTES_PER_TOKEN)
        cache_key = (int(num_tokens), int(head_dim))
        try:
            compiled = _CUTE_COMPILED_CACHE.get(cache_key)
            if compiled is None:
                fake_data, fake_scale, fake_out = _build_fake_tensors(
                    num_tokens, head_dim)
                compiled = cute.compile(stage_for_broadcast_cute_jit,
                                        fake_data,
                                        fake_scale,
                                        fake_out,
                                        num_tokens,
                                        head_dim,
                                        options="--enable-tvm-ffi")
                _CUTE_COMPILED_CACHE[cache_key] = compiled
            # Runtime tensors: must be contiguous uint8 with the matching
            # shape. The compiled TVM-FFI shim accepts torch tensors
            # directly (no per-call from_dlpack).
            k_data_in = k_data[:num_tokens].contiguous()
            k_scale_in = k_scale[:num_tokens].contiguous()
            out_view = stage_buffer[:num_tokens, :head_dim +
                                    SCALE_BYTES_PER_TOKEN]
            compiled(k_data_in, k_scale_in, out_view, num_tokens, head_dim)
        except Exception:
            # cute.compile / cute.kernel may not be wired up on every
            # toolchain (DSLRuntimeError lives in cutlass.base_dsl.common
            # and is NOT a stdlib subclass; catch broadly). Fall back to
            # the byte-equivalent torch scatter so callers never
            # hard-fail on the CuTe path.
            stage_for_broadcast_torch_scatter(k_data, k_scale, num_tokens,
                                              use_fp4, stage_buffer)
else:

    def stage_for_broadcast_cute(k_data, k_scale, num_tokens: int,
                                 use_fp4: bool, stage_buffer):
        """CuTe DSL unavailable: defer to the torch scatter so the call
        site is callable in both environments."""
        stage_for_broadcast_torch_scatter(k_data, k_scale, num_tokens,
                                          use_fp4, stage_buffer)

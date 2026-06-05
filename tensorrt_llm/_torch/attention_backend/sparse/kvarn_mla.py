# SPDX-License-Identifier: Apache-2.0
"""KVarN <-> op-trt MLA latent KV-cache adapter.

Bridges the calibration-free variance-normalized quantizer (``kvarn_core``) to
the DeepSeek-V3.2 MLA latent cache layout used by the op-trt sparse/trtllm
attention path.

MLA latent layout (per token, per the trtllm backend's
``mla_rope_append_paged_kv_assign_q``):

    latent_cache[token] = concat( compressed_kv[kv_lora_rank],  k_pe[qk_rope_head_dim] )

  * ``compressed_kv`` (``ckv``): the learned low-rank latent that the up-proj
    expands into per-head K_nope and V. It is a *per-token vector*; its error
    is dominated by token magnitude (paper Sec 3.1), so KVarN quantizes it in
    the **V orientation** (per-token RTN rows, [group, kv_lora_rank]).
  * ``k_pe``: the RoPE positional key (shared across heads in MLA). Quantized
    in the **K orientation** (per-channel RTN rows, [qk_rope_head_dim, group]).
    Small (typically 64) so its bit savings are minor, but keeping it on the
    same tile keeps one record per (block, ckv+pe).

Tile == one vLLM/op-trt KV block (``tokens_per_block``, must equal the KVarN
``group``; op-trt DSA uses 64-token blocks for the indexer but the MLA main
cache uses ``tokens_per_block`` — the adapter takes ``group`` explicitly and
asserts the caller passes a power-of-two head_dim for the Hadamard).

This adapter is the *component-tier* surface. The *system-tier* wiring
(replacing the fused C++ FP8 quant inside the append op) is described in
``SYSTEM_INTEGRATION`` below and is the production end-state; this Python/Triton
store+dequant is the validated first step that runs on the real latent tensor
before the C++ append, so accuracy and capacity can be measured end to end
without first writing the CUDA op.
"""

from __future__ import annotations

import torch

try:
    from tensorrt_llm._torch.attention_backend.sparse.kvarn_core import (
        hadamard_matrix,
        kvarn_dequant_rows,
        kvarn_quant_rows,
    )
except ImportError:  # standalone / unit-test import (sibling module)
    from kvarn_core import (  # noqa: F401
        hadamard_matrix,
        kvarn_dequant_rows,
        kvarn_quant_rows,
    )

SYSTEM_INTEGRATION = """\
System-tier integration plan (production decode path):

  1. Config flag: set ``sparse_attention_config.mla_latent_kv_dtype`` to a
     ``kvarn_k<ckv>v<pe>`` value and resolve it into a ``KVarNConfig``. This is
     dense MLA latent KV storage only; Indexer K remains controlled by
     ``indexer_k_dtype``.

  2. KV-cache manager: size the MLA latent pool for the packed KVarN record
     (k_bits*kv_lora_rank/8 + v_bits*pe/8 + fp16 scales) per (block) instead of
     fp8's 1 byte/elem. Reserve the fixed fp16 "tail pool" (sink_tokens=128 +
     in-progress tail) per active request/layer (KVarNConfig.pool_bytes) so CUDA
     graphs see a static allocation; cap max_num_seqs to the budget.

  3. Append path: replace the FP8 quant inside
     ``torch.ops.trtllm.mla_rope_append_paged_kv_assign_q``. Two-stage:
       (a) FIRST (validated here): run RoPE+append into an fp16 staging tile;
           when a block fills to ``group`` tokens, call the KVarN store (this
           adapter / the Triton port) and write the packed record. The fused C++
           op runs with quant_mode=fp16 into staging.
       (b) END-STATE: fold the Sinkhorn+RTN store into a C++/CUTLASS variant of
           the append op (kv_scale_orig_quant carries the absorbed scales) so
           there is no extra HBM round-trip, mirroring the fused FP8 path.

  4. Decode path: the MLA generation kernel reads ckv/k_pe from cache. Insert a
     KVarN dequant (Triton ``kvarn_decode`` -> rotated frame -> inverse Hadamard)
     producing fp16 ckv/k_pe that feed the existing BatchMLAPagedAttention decode.
     Fuse the second scale s2 into the dequant kernel (paper App I: <=1.4% over
     single-scale RTN, no extra round-trip).

  5. sink_tokens: the first 128 tokens/request stay fp16 (attention-sink
     preservation). For DeepSeek MLA this is the prompt prefix; keep it in the
     existing fp16 latent path and only KVarN-quantize blocks past the sink.
"""


def quant_latent_block(
    ckv: torch.Tensor,          # [group, kv_lora_rank] fp16/fp32, one full block
    k_pe: torch.Tensor,         # [group, qk_rope_head_dim]
    *,
    ckv_bits: int = 4,
    pe_bits: int = 4,
    iters: int = 16,
    H_ckv: torch.Tensor | None = None,
    H_pe: torch.Tensor | None = None,
) -> dict:
    """Quantize one filled MLA latent block (group tokens).

    ckv uses V-orientation (per-token rows). k_pe uses K-orientation
    (per-channel rows): we transpose to [pe, group] so RTN rows are channels.

    Returns a dict with the two packed records (``ckv_*``, ``kpe_*``) plus the
    Hadamard matrices used (cached by the caller across blocks).
    """
    dev = ckv.device
    G, Dckv = ckv.shape
    _, Dpe = k_pe.shape
    if H_ckv is None:
        H_ckv = hadamard_matrix(Dckv, dev, torch.float32)
    if H_pe is None:
        H_pe = hadamard_matrix(Dpe, dev, torch.float32)

    # ckv: rotate channels, V-orientation -> [1, group, Dckv]
    ckv_rot = (ckv.float() @ H_ckv).unsqueeze(0)
    ckv_rec = kvarn_quant_rows(ckv_rot, ckv_bits, iters)

    # k_pe: rotate channels, then K-orientation [pe, group] -> [1, Dpe, group]
    pe_rot = (k_pe.float() @ H_pe).transpose(0, 1).unsqueeze(0)
    pe_rec = kvarn_quant_rows(pe_rot, pe_bits, iters)

    return {
        "ckv": ckv_rec, "ckv_bits": ckv_bits, "ckv_dim": Dckv,
        "kpe": pe_rec, "pe_bits": pe_bits, "pe_dim": Dpe, "group": G,
        "H_ckv": H_ckv, "H_pe": H_pe,
    }


def dequant_latent_block(rec: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of quant_latent_block -> (ckv [group, Dckv], k_pe [group, Dpe]) fp16."""
    Dckv, Dpe, G = rec["ckv_dim"], rec["pe_dim"], rec["group"]
    ckv_rot = kvarn_dequant_rows(rec["ckv"], rec["ckv_bits"], Dckv).squeeze(0)  # [G, Dckv]
    ckv = (ckv_rot @ rec["H_ckv"]).to(torch.float16)

    pe_rot = kvarn_dequant_rows(rec["kpe"], rec["pe_bits"], G).squeeze(0)        # [Dpe, G]
    k_pe = (pe_rot.transpose(0, 1) @ rec["H_pe"]).to(torch.float16)              # [G, Dpe]
    return ckv, k_pe


def packed_bytes_per_block(Dckv: int, Dpe: int, group: int,
                           ckv_bits: int, pe_bits: int) -> int:
    """Total bytes for one quantized MLA latent block (vs fp16 = 2*group*(Dckv+Dpe))."""
    ckv_data = group * Dckv * ckv_bits // 8
    ckv_scales = (2 * group + Dckv) * 2          # s_row_abs[G]+zp[G]+s_col[Dckv] fp16
    pe_data = group * Dpe * pe_bits // 8
    pe_scales = (2 * Dpe + group) * 2            # K-orient: rows=Dpe
    return ckv_data + ckv_scales + pe_data + pe_scales

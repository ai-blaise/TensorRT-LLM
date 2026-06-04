# SPDX-License-Identifier: Apache-2.0
"""Generate a packed KVarN fixture + fp16 reference for the in-kernel CUDA bench.

Writes, for N blocks of real-shaped MLA latent (group=64, ckv=512, k_pe=64):
  - fixture_store.bin : the packed bytes the C++ KVarNLatentPool would hold,
                        block-contiguous in the EXACT _compute_layout order.
  - fixture_ref.bin   : the fp16 ckv[N,group,512] then k_pe[N,group,64] that the
                        python dequant_latent_block produces (the ground truth the
                        in-kernel kernel must reproduce).
  - fixture_meta.txt  : N and byte sizes.

The CUDA bench reads fixture_store.bin into its store, runs the in-kernel dequant,
and we compare its output to fixture_ref.bin -> cosine. This proves the in-kernel
FWHT-based dequant matches the validated python (matmul-Hadamard) path.
"""
import sys, struct
import numpy as np
import torch

sys.path.insert(0, "tensorrt_llm/_torch/attention_backend/sparse")
from kvarn_core import hadamard_matrix  # noqa: E402
from kvarn_mla import quant_latent_block, dequant_latent_block  # noqa: E402

GROUP, DCKV, DPE = 64, 512, 64
CKV_BITS, PE_BITS = 4, 2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
dev = torch.device("cuda")
torch.manual_seed(7)

H_ckv = hadamard_matrix(DCKV, dev, torch.float32)
H_pe = hadamard_matrix(DPE, dev, torch.float32)


def layout_order(rec_ckv, rec_pe):
    # MUST match BlockOffsets/layout() in the .cu and KVarNLatentPool._compute_layout
    return [
        rec_ckv["q_packed"].reshape(-1).view(torch.uint8),
        rec_ckv["s_row_abs"].reshape(-1).view(torch.uint8),
        rec_ckv["zp_abs"].reshape(-1).view(torch.uint8),
        rec_ckv["s_col"].reshape(-1).view(torch.uint8),
        rec_pe["q_packed"].reshape(-1).view(torch.uint8),
        rec_pe["s_row_abs"].reshape(-1).view(torch.uint8),
        rec_pe["zp_abs"].reshape(-1).view(torch.uint8),
        rec_pe["s_col"].reshape(-1).view(torch.uint8),
    ]


store_blocks = []
ref_ckv = []
ref_kpe = []
cos_acc = []
for b in range(N):
    # realistic latent: low-rank-ish with a few outlier channels (paper regime)
    ckv = torch.randn(GROUP, DCKV, device=dev) * 0.5
    ckv[:, ::37] *= 6.0  # channel outliers
    k_pe = torch.randn(GROUP, DPE, device=dev) * 0.8
    rec = quant_latent_block(ckv, k_pe, ckv_bits=CKV_BITS, pe_bits=PE_BITS,
                             iters=16, H_ckv=H_ckv, H_pe=H_pe)
    parts = layout_order(rec["ckv"], rec["kpe"])
    store_blocks.append(torch.cat(parts).cpu().numpy().astype(np.uint8))
    ckv_dq, kpe_dq = dequant_latent_block(rec)
    ref_ckv.append(ckv_dq.float().cpu().numpy())
    ref_kpe.append(kpe_dq.float().cpu().numpy())
    # sanity: python round-trip cos on ckv
    c = torch.nn.functional.cosine_similarity(
        ckv.flatten().float(), ckv_dq.flatten().float(), dim=0).item()
    cos_acc.append(c)

store = np.stack(store_blocks)  # [N, total_bytes]
ref_ckv = np.stack(ref_ckv).astype(np.float16)  # [N, GROUP, DCKV]
ref_kpe = np.stack(ref_kpe).astype(np.float16)  # [N, GROUP, DPE]

store.tofile("fixture_store.bin")
ref_ckv.tofile("fixture_ref_ckv.bin")
ref_kpe.tofile("fixture_ref_kpe.bin")
with open("fixture_meta.txt", "w") as f:
    f.write(f"N={N}\ntotal_bytes={store.shape[1]}\n")
print(f"wrote fixture: N={N} total_bytes/block={store.shape[1]} "
      f"python_ckv_roundtrip_cos_mean={np.mean(cos_acc):.5f}")

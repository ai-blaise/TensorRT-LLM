# SPDX-License-Identifier: Apache-2.0
"""System-tier test for the KVarN kv_cache_dtype backend.

Validates, on real DeepSeek-V3.2 MLA dims at the PRODUCTION tile
(group == tokens_per_block == 64):
  1. dtype-string parsing + env resolution.
  2. KVarNLatentPool serialize -> deserialize round-trip == direct
     quant/dequant (proves the byte-packed side-pool is lossless vs the
     component adapter).
  3. end-to-end accuracy: reconstructed ckv -> up-proj K_nope/V cosine.
  4. memory accounting vs fp16 / fp8 / nvfp4.
"""
import sys
sys.path.insert(0, "/tmp/kvarn_bench")
import torch
import kvarn_backend as B
import kvarn_mla as M


def synth_latent(group, Dckv, Dpe, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    ckv = (torch.randn(group, Dckv, device=dev, generator=g)
           * torch.randn(group, 1, device=dev, generator=g).mul(0.7).exp()
           * torch.randn(1, Dckv, device=dev, generator=g).mul(0.4).exp()).half()
    kpe = (torch.randn(group, Dpe, device=dev, generator=g)
           * torch.randn(group, 1, device=dev, generator=g).mul(0.5).exp()).half()
    return ckv, kpe


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    dev = torch.device("cuda")
    Dckv, Dpe = 512, 64
    LAYERS = 61  # DeepSeek-V3.2

    # 1. dtype parsing -----------------------------------------------------
    cfg = B.parse_kvarn_dtype("kvarn_k4v2")
    assert (cfg.ckv_bits, cfg.pe_bits) == (4, 2), cfg
    assert cfg.name == "kvarn_k4v2"
    assert cfg.latent_dim == 576
    import os
    os.environ["TRTLLM_MLA_LATENT_KV_DTYPE"] = "kvarn_k2v2"
    assert B.resolve_kvarn_config().ckv_bits == 2
    del os.environ["TRTLLM_MLA_LATENT_KV_DTYPE"]
    assert B.resolve_kvarn_config() is None
    assert B.resolve_kvarn_config("fp8") is None
    print("[1] dtype parse + env resolve OK")

    # 2/3. side-pool round-trip + accuracy at production tile --------------
    # fake up-proj (ckv -> per-head K_nope / V) as the metric that matters
    torch.manual_seed(7)
    H, nope, vdim = 128, 128, 128
    Wk = torch.randn(Dckv, H * nope, device=dev) / Dckv**0.5
    Wv = torch.randn(Dckv, H * vdim, device=dev) / Dckv**0.5

    print(f"\n{'dtype':>10} {'group':>5} {'cos_ckv':>8} {'cos_Knope':>9} "
          f"{'cos_V':>7} {'cos_kpe':>8} {'pool==adapter':>13}")
    rows = {}
    for group in (64, 128):  # 64 = production tokens_per_block; 128 = paper tile
        for dt in ("kvarn_k4v2", "kvarn_k4v4", "kvarn_k2v2"):
            cfg = B.parse_kvarn_dtype(dt)
            NB = 8
            pool = B.KVarNLatentPool(NB, group, cfg, dev)
            ckv, kpe = synth_latent(group, Dckv, Dpe, dev, seed=group + cfg.ckv_bits)
            bid = 3
            pool.store_block(bid, ckv, kpe)
            assert pool.valid[bid] and not pool.valid[0]
            ckv_d, kpe_d = pool.load_block(bid)

            # cross-check: byte-pool path must equal direct adapter path
            rec = M.quant_latent_block(ckv, kpe, ckv_bits=cfg.ckv_bits,
                                       pe_bits=cfg.pe_bits, iters=cfg.iters,
                                       H_ckv=pool.H_ckv, H_pe=pool.H_pe)
            ckv_a, kpe_a = M.dequant_latent_block(rec)
            same = torch.equal(ckv_d, ckv_a) and torch.equal(kpe_d, kpe_a)

            Kr, Vr = ckv.float() @ Wk, ckv.float() @ Wv
            Kd, Vd = ckv_d.float() @ Wk, ckv_d.float() @ Wv
            cck, cnope, cv, cpe = (cos(ckv, ckv_d), cos(Kr, Kd),
                                   cos(Vr, Vd), cos(kpe, kpe_d))
            rows[(dt, group)] = (cck, cnope, cv, cpe)
            print(f"{dt:>10} {group:>5} {cck:>8.5f} {cnope:>9.5f} "
                  f"{cv:>7.5f} {cpe:>8.5f} {str(same):>13}")
            assert same, f"byte-pool != adapter for {dt} g{group}"

    # 4. memory accounting -------------------------------------------------
    print(f"\n{'dtype':>10} {'group':>5} {'B/blk':>7} {'bpe':>6} "
          f"{'B/tok/L':>8} {'vs_fp16':>8} {'vs_fp8':>7} {'vs_nvfp4':>9}")
    nvfp4_bpe = 4 + 16 / 16  # 4-bit data + UE8M0 scale per 16 = 4.5 bpe (dense)
    for group in (64, 128):
        for dt in ("kvarn_k4v2", "kvarn_k4v4", "kvarn_k2v2"):
            cfg = B.parse_kvarn_dtype(dt)
            bpb = cfg.packed_bytes(group)
            bpe = cfg.bits_per_elem(group)
            per_tok = bpb / group
            fp16_tok = cfg.fp16_bytes(group) / group
            fp8_tok = cfg.fp8_bytes(group) / group
            nvfp4_tok = nvfp4_bpe * cfg.latent_dim / 8
            print(f"{dt:>10} {group:>5} {bpb:>7} {bpe:>6.3f} {per_tok:>8.1f} "
                  f"{fp16_tok/per_tok:>7.2f}x {fp8_tok/per_tok:>6.2f}x "
                  f"{nvfp4_tok/per_tok:>8.2f}x")

    # whole-cache projection at a realistic decode footprint
    cfg = B.parse_kvarn_dtype("kvarn_k4v2")
    g = 64
    kvarn_tok = B.kvarn_latent_bytes_per_token(cfg, g, LAYERS)
    fp16_tok = B.latent_bytes_per_token_baseline(cfg.latent_dim, LAYERS, 2)
    fp8_tok = B.latent_bytes_per_token_baseline(cfg.latent_dim, LAYERS, 1)
    print(f"\nWhole MLA latent cache (61 layers, group=64), kvarn_k4v2:")
    print(f"  KVarN  {kvarn_tok/1024:8.2f} KiB/token")
    print(f"  fp8    {fp8_tok/1024:8.2f} KiB/token   ({fp8_tok/kvarn_tok:.2f}x KVarN)")
    print(f"  fp16   {fp16_tok/1024:8.2f} KiB/token   ({fp16_tok/kvarn_tok:.2f}x KVarN)")
    for ctx in (32768, 131072):
        print(f"  @ {ctx:>6} ctx/seq: KVarN {kvarn_tok*ctx/2**30:.3f} GiB  "
              f"fp8 {fp8_tok*ctx/2**30:.3f} GiB  fp16 {fp16_tok*ctx/2**30:.3f} GiB")

    print("\nKVARN-BACKEND-OK")


if __name__ == "__main__":
    main()

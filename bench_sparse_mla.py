"""Standalone correctness+timing harness for trtllm.sparse_mla_decode_nvfp4.

Validation model (per task NOTE 2): the optimized kernel must be numerically
equivalent to the REFERENCE kernel on the SAME inputs. The unmodified built op
is the reference; a modified build is compared against the saved baseline
(O + LSE). This script can either SAVE a baseline or COMPARE against one, and
always reports kernel us via CUDA events.
"""
import argparse, os, time, json, ctypes
import torch

# Register torch.ops.trtllm.* by loading the C++ extension directly, avoiding
# the heavy `import tensorrt_llm` (which pulls a deep_gemm overlay chain).
_SP = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm"
for _so in (_SP + "/libs/libtensorrt_llm.so", _SP + "/libs/libth_common.so"):
    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
assert hasattr(torch.ops.trtllm, "sparse_mla_decode_nvfp4"), "op not registered"

torch.manual_seed(0)

H_Q = 128
D_QK = 576          # kv_lora_rank(512) + qk_rope(64)
D_V = 512
PAGE = 64
KV_BYTES = 288      # 576 fp4 / 2
SCALE_BYTES = 36    # one e4m3 per 16 elems, padded layout in pool is 36


def make_inputs(b, s_q, topk, num_pages, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, s_q, H_Q, D_QK, dtype=torch.bfloat16, device=device, generator=g)
    # Random packed NVFP4 bytes + e4m3 scale bytes. Values are arbitrary but
    # FIXED by seed so baseline and candidate see identical bits.
    kv = torch.randint(0, 256, (num_pages, PAGE, 1, KV_BYTES), dtype=torch.uint8, device=device, generator=g)
    kv_scales = torch.randint(0, 256, (num_pages, PAGE, 1, SCALE_BYTES), dtype=torch.uint8, device=device, generator=g)
    # Selected token indices into the flattened (page*PAGE) token space.
    max_tok = num_pages * PAGE
    idx = torch.randint(0, max_tok, (b, s_q, topk), dtype=torch.int32, device=device, generator=g)
    # Some -1 padding (invalid) entries, like real topk.
    pad_mask = torch.rand(b, s_q, topk, generator=g, device=device) < 0.1
    idx = idx.masked_fill(pad_mask, -1)
    return q, kv, kv_scales, idx


def run_once(q, kv, kv_scales, idx, sm_scale):
    return torch.ops.trtllm.sparse_mla_decode_nvfp4(
        q, kv, kv_scales, idx, d_v=D_V, sm_scale=sm_scale)


def bench(q, kv, kv_scales, idx, sm_scale, iters=50, warmup=10):
    for _ in range(warmup):
        run_once(q, kv, kv_scales, idx, sm_scale)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        run_once(q, kv, kv_scales, idx, sm_scale)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=32)
    ap.add_argument("--s_q", type=int, default=1)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--pages", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--compare", type=str, default="")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda"
    sm_scale = 1.0 / (D_QK ** 0.5)

    q, kv, kv_scales, idx = make_inputs(args.b, args.s_q, args.topk, args.pages, dev, args.seed)
    out = run_once(q, kv, kv_scales, idx, sm_scale)
    o, lse = out[0], out[1]
    torch.cuda.synchronize()

    us = bench(q, kv, kv_scales, idx, sm_scale, iters=args.iters)
    shape = dict(b=args.b, s_q=args.s_q, topk=args.topk, pages=args.pages)
    print(f"SHAPE {json.dumps(shape)}  kernel_us={us:.2f}  o={tuple(o.shape)} lse={tuple(lse.shape)}")

    if args.save:
        torch.save({"o": o.detach().cpu(), "lse": lse.detach().cpu(), "shape": shape, "us": us}, args.save)
        print(f"SAVED baseline -> {args.save}")
    if args.compare:
        ref = torch.load(args.compare, map_location="cpu")
        ro, rl = ref["o"].cuda(), ref["lse"].cuda()
        # O is bf16; LSE is fp32. Numerical-equivalence check.
        o_max = (o.float() - ro.float()).abs().max().item()
        o_mean = (o.float() - ro.float()).abs().mean().item()
        # finite mask for LSE (invalid rows can be -inf)
        fm = torch.isfinite(lse) & torch.isfinite(rl)
        l_max = (lse[fm] - rl[fm]).abs().max().item() if fm.any() else 0.0
        bit_o = torch.equal(o, ref["o"].cuda())
        bit_l = torch.equal(lse, ref["lse"].cuda())
        print(f"COMPARE o_max_abs={o_max:.3e} o_mean_abs={o_mean:.3e} lse_max_abs={l_max:.3e} "
              f"bit_identical_o={bit_o} bit_identical_lse={bit_l} "
              f"baseline_us={ref['us']:.2f} delta_us={us-ref['us']:+.2f}")


if __name__ == "__main__":
    main()

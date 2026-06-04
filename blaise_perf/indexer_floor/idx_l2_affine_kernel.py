"""L2 affine-reuse via a single fused CUDA kernel (load_inline).

The remap convert_req_index_to_global is a gather (block_table lookup) +
affine. On an S-layer the local topk == F-layer's, so the global indices are
F's global + (dLayer)*blockSize on valid entries (-1 stays -1). A single
fused elementwise kernel does this in one launch -- much cheaper than re-
running the gather, and (the test below) than torch.where which launches
several kernels. This is the real lever-2 candidate.

Compares, graphed at prod shape:
  full remap kernel (baseline per-layer)  vs
  fused affine kernel (S-layer reuse)     vs
  torch.where affine (the naive reuse)
plus a numerical-equality correctness check against the full remap.
"""
import torch
from torch.utils.cpp_extension import load_inline

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401

DEV = torch.device("cuda")

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void affine_reuse_kernel(const int* __restrict__ g, int* __restrict__ out,
                                    long n, int delta) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = g[i];
    out[i] = (v < 0) ? -1 : v + delta;
}

torch::Tensor affine_reuse(torch::Tensor g, int64_t delta) {
    auto out = torch::empty_like(g);
    long n = g.numel();
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    affine_reuse_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        g.data_ptr<int>(), out.data_ptr<int>(), n, (int)delta);
    return out;
}
"""
CPP = "torch::Tensor affine_reuse(torch::Tensor g, int64_t delta);"

m = load_inline(name="idx_affine_reuse", cpp_sources=CPP, cuda_sources=CUDA,
                functions=["affine_reuse"], verbose=False)


@torch.inference_mode()
def bench(fn, iters=500, warmup=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


@torch.inference_mode()
def graphed(fn, iters=500, warmup=80):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return bench(lambda: g.replay(), iters, warmup)


def main():
    M, topk, block, NL = 8, 1024, 64, 58
    prefix = 4608
    stride = NL * block
    nblk = (prefix + block - 1) // block + 1
    req = torch.arange(M, device=DEV, dtype=torch.int32)
    bt = torch.arange(M * nblk, device=DEV, dtype=torch.int32).view(M, nblk)
    tok = torch.randint(0, prefix, (M, topk), device=DEV, dtype=torch.int32)
    tok[:, -(topk // 8):] = -1
    op = torch.ops.trtllm.convert_req_index_to_global
    lF, lS = 4, 5
    delta = (lS - lF) * block

    g_F = op(req, bt, tok, block, topk, stride, lF).contiguous()
    g_S_ref = op(req, bt, tok, block, topk, stride, lS)

    out_k = m.affine_reuse(g_F, delta)
    ok_kernel = bool(torch.equal(out_k, g_S_ref))
    out_w = torch.where(g_F < 0, g_F, g_F + delta)
    ok_where = bool(torch.equal(out_w, g_S_ref))

    print(f"# device={torch.cuda.get_device_name(0)} M={M} topk={topk} elems={M*topk}")
    print(f"# kernel correct={ok_kernel} where correct={ok_where} delta={delta}")

    t_full_g = graphed(lambda: op(req, bt, tok, block, topk, stride, lS))
    t_kern_g = graphed(lambda: m.affine_reuse(g_F, delta))
    t_where_g = graphed(lambda: torch.where(g_F < 0, g_F, g_F + delta))
    print("=== RESULTS_US ===")
    print(f"L2_remap_full_graphed_us\t{t_full_g:.3f}")
    print(f"L2_affine_fused_kernel_graphed_us\t{t_kern_g:.3f}")
    print(f"L2_affine_where_graphed_us\t{t_where_g:.3f}")
    save = (t_full_g - t_kern_g) * 43 / 1000.0
    print(f"L2_fused_kernel_save_43S_ms\t{save:.4f}")
    print("=== END ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# PDE two-level fused top-k — iteration 1.
#
# Goal: a SINGLE device-resident fused kernel that does the prod HISA two-level
# top-k in one launch with on-device control flow, beating the faithful prod
# op-sequence baseline while staying correct (>=99% per-row set-match vs a torch
# reference that implements the SAME two-level algorithm).
#
# Prod semantics (from dsa.py _hisa_topk_from_logits / _hisa_topk_from_nvfp4_cache):
#   (1) block score = amax over each block_size(=128)-token block, invalid (>=seq_len)
#       and padding tokens = -inf.
#   (2) radix-select top block_topk(=64) blocks by block score.
#   (3) candidate token scores = tokens of the selected blocks (block_topk*block_size).
#   (4) radix-select top final_topk(=1024) candidate tokens.
#   (5) output GLOBAL token indices [B, final_topk] int32, descending score,
#       short rows padded with -1.
#
# Reuses the EXACT composite-key radix-select logic from
# /host_repo/cpp/tensorrt_llm/kernels/pde/pde_g3_dev_ctrl.cuh (make_key64,
# float_to_okey, the MSD radix-select), adapted for the [B,S] token layout and
# the two-level structure.

import os, sys, time, math
import torch

CUDA_SRC = r'''
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

namespace cg = cooperative_groups;

// ---- composite-key radix helpers (ported verbatim from pde_g3_dev_ctrl.cuh) ----
__device__ __forceinline__ unsigned int float_to_okey(float f) {
  unsigned int u = __float_as_uint(f);
  unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
  return u ^ mask;
}
__device__ __forceinline__ unsigned long long make_key64(float score, int idx) {
  unsigned int sk = float_to_okey(score);
  unsigned int ik = ~(unsigned int)idx;          // smaller idx -> larger ik
  return ((unsigned long long)sk << 32) | (unsigned long long)ik;
}

// generic scalar->float load (fp32 passthrough, bf16 convert)
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

constexpr int kRadixBits = 8;
constexpr int kRadixBins = 1 << kRadixBits;     // 256
constexpr int kRadixPasses = 64 / kRadixBits;   // 8
constexpr int kThreads = 256;

// =====================================================================
// FUSED two-level top-k, ONE cooperative launch.
//   grid: one CTA per row (B CTAs). blockDim = 256.
//   Stage A (per-CTA, no grid sync needed since each row is independent):
//     A1  compute block scores (amax) into g_block_scores[row, :]
//     A2  radix-select top block_topk blocks -> mark g_block_sel[row, blk]=1
//     A3  radix-select top final_topk tokens AMONG selected blocks ->
//         emit global token indices (descending) into out_idx[row,:]
//
// Because each row is fully independent and handled by exactly one CTA, NO
// grid.sync is required: the whole two-level pipeline is __syncthreads-only
// within the CTA. (Cooperative launch kept for API uniformity / future
// multi-CTA-per-row; here ctas==B.)
//
// SMEM histogram radix-select operates over a logical candidate set defined by
// a predicate (active iff key's fixed high bits == prefix). Level 1's set =
// all valid blocks. Level 2's set = all tokens whose block is selected AND
// token < seq_len. We fuse the gather away: stage A3 never materializes the
// candidate score array — it streams tokens, tests block-membership via
// g_block_sel, and histograms directly.
// =====================================================================

template <typename ScalarT>
__global__ void __launch_bounds__(kThreads) fused_two_level_topk(
    const ScalarT* __restrict__ scores,   // [B, S]
    const int* __restrict__ seq_lens,     // [B]
    int* __restrict__ out_idx,            // [B, final_topk]
    float* __restrict__ g_block_scores,   // [B, num_blocks] scratch
    unsigned char* __restrict__ g_block_sel, // [B, num_blocks] scratch (0/1)
    int B, int S, int num_blocks,
    int block_size, int block_topk, int final_topk) {

  const int row = blockIdx.x;
  if (row >= B) return;
  const int tid = threadIdx.x;
  const int seq_len = seq_lens[row];

  const ScalarT* srow = scores + (size_t)row * S;
  float* bscore = g_block_scores + (size_t)row * num_blocks;
  unsigned char* bsel = g_block_sel + (size_t)row * num_blocks;
  int* orow = out_idx + (size_t)row * final_topk;

  __shared__ unsigned int s_hist[kRadixBins];

  // ---------- A1: block scores = amax over each block, -inf for invalid ----------
  // each block handled by a strided set of threads; reduce per block.
  // Simple: thread loops over blocks (grid-stride within CTA), reduces its block.
  for (int b = tid; b < num_blocks; b += blockDim.x) {
    int start = b * block_size;
    int end = min(start + block_size, seq_len);
    float m = -FLT_MAX;
    for (int t = start; t < end; ++t) {
      float v = to_float(srow[t]);
      m = fmaxf(m, v);
    }
    bscore[b] = m;          // blocks fully past seq_len get -FLT_MAX
    bsel[b] = 0;
  }
  __syncthreads();

  // number of valid blocks for this row
  int valid_blocks = (seq_len + block_size - 1) / block_size;
  int k_blocks = min(block_topk, valid_blocks);

  // ---------- A2: radix-select top k_blocks blocks ----------
  // composite key over block scores; select set { key >= threshold }.
  // single-CTA MSD radix-select using SMEM histogram.
  {
    unsigned long long prefix = 0ull, prefix_mask = 0ull;
    int k_remain = k_blocks;
    for (int t = 0; t < kRadixPasses; ++t) {
      const int shift = 64 - kRadixBits * (t + 1);
      for (int bn = tid; bn < kRadixBins; bn += blockDim.x) s_hist[bn] = 0u;
      __syncthreads();
      for (int b = tid; b < valid_blocks; b += blockDim.x) {
        unsigned long long k = make_key64(bscore[b], b);
        if ((k & prefix_mask) == prefix) {
          unsigned int d = (unsigned int)((k >> shift) & (kRadixBins - 1));
          atomicAdd(&s_hist[d], 1u);
        }
      }
      __syncthreads();
      // every thread walks identical histogram (already in SMEM) -> identical prefix.
      // do the cumulative walk in ONE warp to avoid 256x redundant scans? keep simple:
      // serialize over thread 0 result via shared.
      __shared__ int s_digit, s_acc;
      if (tid == 0) {
        int acc = 0, digit = 0;
        for (int bn = kRadixBins - 1; bn >= 0; --bn) {
          int cnt = (int)s_hist[bn];
          if (acc + cnt >= k_remain) { digit = bn; break; }
          acc += cnt;
        }
        s_digit = digit; s_acc = acc;
      }
      __syncthreads();
      k_remain -= s_acc;
      prefix |= ((unsigned long long)s_digit) << shift;
      prefix_mask |= ((unsigned long long)(kRadixBins - 1)) << shift;
      __syncthreads();
    }
    // mark selected blocks: { key >= prefix }
    const unsigned long long thr = prefix;
    for (int b = tid; b < valid_blocks; b += blockDim.x) {
      unsigned long long k = make_key64(bscore[b], b);
      if (k >= thr) bsel[b] = 1;
    }
  }
  __syncthreads();

  // ---------- A3: radix-select top final_topk tokens among selected blocks ----------
  // candidate set = { token t : t < seq_len AND bsel[t / block_size] == 1 }.
  // emit GLOBAL token index. Gather fused away (no candidate array).
  int cand_count = k_blocks * block_size;   // upper bound on candidate tokens
  int k_tok = min(final_topk, cand_count);
  // also bound by valid tokens in selected blocks; if fewer, pad with -1.
  {
    unsigned long long prefix = 0ull, prefix_mask = 0ull;
    int k_remain = k_tok;
    for (int t = 0; t < kRadixPasses; ++t) {
      const int shift = 64 - kRadixBits * (t + 1);
      for (int bn = tid; bn < kRadixBins; bn += blockDim.x) s_hist[bn] = 0u;
      __syncthreads();
      for (int tok = tid; tok < seq_len; tok += blockDim.x) {
        int blk = tok / block_size;
        if (bsel[blk]) {
          unsigned long long k = make_key64(to_float(srow[tok]), tok);
          if ((k & prefix_mask) == prefix) {
            unsigned int d = (unsigned int)((k >> shift) & (kRadixBins - 1));
            atomicAdd(&s_hist[d], 1u);
          }
        }
      }
      __syncthreads();
      __shared__ int s_digit2, s_acc2;
      if (tid == 0) {
        int acc = 0, digit = 0;
        for (int bn = kRadixBins - 1; bn >= 0; --bn) {
          int cnt = (int)s_hist[bn];
          if (acc + cnt >= k_remain) { digit = bn; break; }
          acc += cnt;
        }
        s_digit2 = digit; s_acc2 = acc;
      }
      __syncthreads();
      k_remain -= s_acc2;
      prefix |= ((unsigned long long)s_digit2) << shift;
      prefix_mask |= ((unsigned long long)(kRadixBins - 1)) << shift;
      __syncthreads();
    }
    // emit: tokens with key >= threshold, into out_idx in descending order.
    // We need DESCENDING score order. The radix threshold gives the SET; to emit
    // in order we use the composite key's ordering: assign each selected token a
    // rank = number of selected tokens with key strictly greater. That's O(K^2)
    // naively; instead we do a simple approach: atomic append then it's unordered.
    // For set-match correctness order doesn't matter, but spec wants descending.
    // Iteration 1: emit via atomic counter (UNORDERED), fill rest with -1.
    __shared__ unsigned int s_fill;
    if (tid == 0) s_fill = 0u;
    __syncthreads();
    const unsigned long long thr = prefix;
    for (int tok = tid; tok < seq_len; tok += blockDim.x) {
      int blk = tok / block_size;
      if (bsel[blk]) {
        unsigned long long k = make_key64(to_float(srow[tok]), tok);
        if (k >= thr) {
          unsigned int pos = atomicAdd(&s_fill, 1u);
          if (pos < (unsigned int)final_topk) orow[pos] = tok;
        }
      }
    }
    __syncthreads();
    // pad remaining with -1
    unsigned int filled = s_fill;
    for (int j = filled + tid; j < final_topk; j += blockDim.x) orow[j] = -1;
  }
}

// ---------------- host launchers ----------------
void launch_fused(torch::Tensor scores, torch::Tensor seq_lens,
                  torch::Tensor out_idx, torch::Tensor block_scores,
                  torch::Tensor block_sel, int block_size, int block_topk,
                  int final_topk) {
  int B = scores.size(0);
  int S = scores.size(1);
  int num_blocks = (S + block_size - 1) / block_size;
  dim3 grid(B), block(kThreads);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (scores.scalar_type() == torch::kFloat32) {
    fused_two_level_topk<float><<<grid, block, 0, stream>>>(
        scores.data_ptr<float>(), seq_lens.data_ptr<int>(),
        out_idx.data_ptr<int>(), block_scores.data_ptr<float>(),
        (unsigned char*)block_sel.data_ptr<uint8_t>(),
        B, S, num_blocks, block_size, block_topk, final_topk);
  } else {
    fused_two_level_topk<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)scores.data_ptr<at::BFloat16>(),
        seq_lens.data_ptr<int>(),
        out_idx.data_ptr<int>(), block_scores.data_ptr<float>(),
        (unsigned char*)block_sel.data_ptr<uint8_t>(),
        B, S, num_blocks, block_size, block_topk, final_topk);
  }
}
'''

def build():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext_pde")
    os.makedirs("/tmp/torch_ext_pde", exist_ok=True)
    t0 = time.time()
    cpp_decl = (
        "#include <torch/extension.h>\n"
        "void launch_fused(torch::Tensor scores, torch::Tensor seq_lens,\n"
        "                  torch::Tensor out_idx, torch::Tensor block_scores,\n"
        "                  torch::Tensor block_sel, int block_size, int block_topk,\n"
        "                  int final_topk);\n"
    )
    mod = load_inline(
        name="pde_topk_v1",
        cpp_sources=cpp_decl,
        cuda_sources=CUDA_SRC,
        functions=["launch_fused"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
        verbose=True,
    )
    print(f"[build] JIT compiled in {time.time()-t0:.1f}s", flush=True)
    return mod


# --------- torch reference: SAME two-level algorithm ---------
def torch_ref_two_level(scores, seq_lens, block_size, block_topk, final_topk):
    B, S = scores.shape
    num_blocks = (S + block_size - 1) // block_size
    pad = num_blocks * block_size - S
    sc = scores.float()
    cols = torch.arange(S, device=scores.device)
    valid = cols.unsqueeze(0) < seq_lens.unsqueeze(1)
    sc = sc.masked_fill(~valid, float("-inf"))
    if pad:
        sc = torch.nn.functional.pad(sc, (0, pad), value=float("-inf"))
    block_scores = sc.reshape(B, num_blocks, block_size).amax(dim=-1)  # [B, nb]
    # top block_topk blocks
    bt = min(block_topk, num_blocks)
    block_ids = block_scores.topk(bt, dim=-1, sorted=False)[1]  # [B, bt]
    offsets = torch.arange(block_size, device=scores.device)
    sel_idx = (block_ids.unsqueeze(-1) * block_size + offsets).reshape(B, -1)  # [B, bt*bs]
    sel_scores = sc.gather(1, sel_idx)  # [B, bt*bs]
    ft = min(final_topk, sel_idx.shape[1])
    rel = sel_scores.topk(ft, dim=-1, sorted=False)[1]  # [B, ft]
    glob = sel_idx.gather(1, rel.long())  # [B, ft] global token idx
    # mask invalid (past seq_len or padding) -> -1
    glob = glob.masked_fill(glob >= seq_lens.unsqueeze(1), -1)
    # pad to final_topk with -1
    if ft < final_topk:
        padt = torch.full((B, final_topk - ft), -1, dtype=glob.dtype, device=glob.device)
        glob = torch.cat([glob, padt], dim=1)
    return glob.to(torch.int32)


def set_match(a, b):
    # a, b: [B, K] int32, -1 = pad. per-row set match fraction (ignoring -1).
    B = a.shape[0]
    fracs = []
    for r in range(B):
        sa = set(x for x in a[r].tolist() if x >= 0)
        sb = set(x for x in b[r].tolist() if x >= 0)
        if len(sb) == 0:
            fracs.append(1.0 if len(sa) == 0 else 0.0)
        else:
            fracs.append(len(sa & sb) / len(sb))
    return sum(fracs) / len(fracs), min(fracs)


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    torch.manual_seed(0)
    mod = build()

    block_size, block_topk, final_topk = 128, 64, 1024

    shapes = []
    for B in [1, 8, 32, 64]:
        for S in [16384, 65536, 132096]:
            shapes.append((B, S))

    print("\n=== correctness + fused timing ===", flush=True)
    results = []
    for (B, S) in shapes:
        scores = torch.randn(B, S, device=dev, dtype=torch.float32)
        # seq_lens: full length (prod decode: uniform, all rows ~max). Use S.
        seq_lens = torch.full((B,), S, device=dev, dtype=torch.int32)
        num_blocks = (S + block_size - 1) // block_size
        out_idx = torch.empty(B, final_topk, device=dev, dtype=torch.int32)
        block_scores = torch.empty(B, num_blocks, device=dev, dtype=torch.float32)
        block_sel = torch.empty(B, num_blocks, device=dev, dtype=torch.uint8)

        # reference
        ref = torch_ref_two_level(scores, seq_lens, block_size, block_topk, final_topk)

        # fused
        mod.launch_fused(scores, seq_lens, out_idx, block_scores, block_sel,
                         block_size, block_topk, final_topk)
        torch.cuda.synchronize()

        avg, mn = set_match(out_idx, ref)

        # timing fused
        for _ in range(15):
            mod.launch_fused(scores, seq_lens, out_idx, block_scores, block_sel,
                             block_size, block_topk, final_topk)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50):
            mod.launch_fused(scores, seq_lens, out_idx, block_scores, block_sel,
                             block_size, block_topk, final_topk)
        en.record(); torch.cuda.synchronize()
        fused_us = st.elapsed_time(en) / 50 * 1000.0

        print(f"B={B:3d} S={S:6d}  fused={fused_us:8.2f}us  set-match avg={avg:.4f} min={mn:.4f}", flush=True)
        results.append((B, S, fused_us, avg, mn))

    print("\n=== summary ===", flush=True)
    for (B, S, f, a, m) in results:
        print(f"B={B:3d} S={S:6d}  fused={f:8.2f}us  setmatch_avg={a:.4f} setmatch_min={m:.4f}")


if __name__ == "__main__":
    main()

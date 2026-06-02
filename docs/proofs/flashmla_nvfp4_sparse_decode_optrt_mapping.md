# FlashMLA NVFP4 Sparse MLA Decode op-trt Mapping

## Reference

The native op-trt NVFP4 sparse MLA decode path is a direct, reference-preserving
port of `ai-blaise/FlashMLA` branch `ai-blaise/nvfp4-kv-decode` at commit
`a2e19e03ecf418cecf25f270caba4ff3196a0d9e`.

The porting rule is intentionally strict: copy the FlashMLA kernel/support
source as-is where possible, map only the op-trt interface boundary, then optimize
adapter policy or native op-trt integration only after source parity and focused
B200 tests are green.

The source parity gate is:

```bash
python3 benchmarks/python/check_flashmla_nvfp4_source_parity.py   --flashmla-csrc /path/to/FlashMLA/csrc   --optrt-nvfp4-sparse cpp/tensorrt_llm/kernels/flashMLA/nvfp4_sparse
```

It compares the copied FlashMLA files byte-for-byte after the op-trt include-root
remap. The copied files include the V3.2 head64 NVFP4 decode kernel, scheduler
metadata kernel, combine header/source, params/utils, and the copied `kerutils`
dependency files. The only allowed copied-source body delta remains in
`smxx/decode/combine/combine.cu`: op-trt adds dispatch buckets through 1024
splits and passes dynamic shared-memory size into `cudaLaunchKernelEx`. The
producer math is copied.

The import intentionally omits unrelated FlashMLA surfaces: API bindings, the
vendored CUTLASS tree, head128/head64 BF16 decode, prefill, sm90, model1, and
q_prequant. op-trt owns the build/runtime boundary; this path targets only sparse
MLA NVFP4 decode for DeepSeek-V3.2 single-node B200.

## op-trt Mapping

`cpp/tensorrt_llm/kernels/flashMLA/sparse_mla_decode_nvfp4.{h,cu}` maps op-trt
tensors/strides to FlashMLA `SparseAttnDecodeParams`.

`cpp/tensorrt_llm/thop/SparseMlaDecodeNvfp4Op.cpp` exposes the torch op:

```text
trtllm::sparse_mla_decode_nvfp4(
  q, kv, kv_scales, indices,
  topk_length=None, attn_sink=None,
  tile_scheduler_metadata=None, num_splits=None,
  d_v=512, sm_scale=1.0)
```

Validated target contract:

| field | value |
|---|---:|
| `q` | `[B, 1, 128, 576]`, BF16 |
| `kv` | `[num_pages, 64, 1, 288]`, packed NVFP4 bytes |
| `kv_scales` | `[num_pages, 64, 1, 36]`, E4M3 scale bytes |
| `indices` | `[B, 1, topk]`, int32 token ids |
| `d_v` | 512 |
| page size | 64 |
| target topk | 1024 |

The wrapper computes scheduler metadata when not supplied, launches the copied
FlashMLA head64 NVFP4 producer twice for head ranges `[0, 64)` and `[64, 128)`,
then launches the copied FlashMLA combine kernel. If metadata/splits are supplied,
the supplied metadata shape owns `num_sm_parts`; the adapter does not re-query
scheduler shape/device properties for precomputed metadata.

There is no FP8 or BF16 KV fallback in this path. The KV payload remains NVFP4
packed data plus split E4M3 scale bytes from the op-trt KV manager boundary into
the FlashMLA producer.

## Scheduler Policy

The direct FlashMLA API default is `max(num_sms / s_q, 1)`. That path is finite
for B1/B4 but produces all-NaN output for random mixed packed-FP4 payloads at
B8/B16/B32 under the target V3.2 shape. The op-trt adapter therefore keeps the
FlashMLA SM floor but raises `num_sm_parts` enough to keep each producer split to
one 64-token top-k block for B8+.

Current policy:

- `topk_blocks = ceil(topk / 64)`.
- B8+ producer parts use `B * (topk_blocks + 15)`, capped at 4096.
- B1/B4 use the FlashMLA combine bucket because the direct reference is finite
  and fastest there.
- B8+ combine uses the per-request top-k split bound; producer parts may be much
  larger than combine's required per-request split count.

This is an adapter correctness/performance policy, not a rewrite of the copied
FlashMLA producer.

## Verification

Focused gates were run on `a4-us-002-rl9` with the B200 buildtools image and the
narrow sparse-kernel/torch-extension harness, not a full container rebuild.

Source/build gates:

| gate | result |
|---|---|
| copied-source parity vs FlashMLA `a2e19e0` | pass |
| op-trt CMake target `flash_mla_nvfp4_sparse_src` | pass |
| `git diff --check` | pass |

Same-input direct FlashMLA comparison for finite reference cases:

| B | topk | FlashMLA median us | op-trt native median us | output compare | result |
|---:|---:|---:|---:|---|---|
| 1 | 1024 | 30.781 | 30.776 | max_abs 0, rms 0 | op-trt slight win |
| 4 | 1024 | 30.862 | 30.861 | max_abs 0, rms 0 | op-trt slight win |

Native op-trt sweep after the direct-port adapter policy:

| B | topk | scheduler parts | split count | finite | median us |
|---:|---:|---:|---:|---|---:|
| 1 | 1024 | 148 | 16 | yes | 30.790 |
| 4 | 1024 | 148 | 64 | yes | 31.221 |
| 8 | 1024 | 248 | 128 | yes | 32.876 |
| 16 | 1024 | 496 | 256 | yes | 57.533 |
| 32 | 1024 | 992 | 512 | yes | 118.817 |
| 64 | 1024 | 1984 | 1024 | yes | 236.524 |
| 128 | 1024 | 3968 | 2048 | yes | 466.570 |

B64/B128 random multi-seed gate: 4/4 finite for both shapes, no NaNs, split
histograms `{1:1024}` and `{1:2048}` respectively.

Reference API behavior under the same random mixed packed-FP4 style payloads:
B1/B4 finite; B8/B16/B32 all-NaN with the direct FlashMLA scheduler default. The
op-trt adapter therefore beats the direct reference on correctness for B8+ and
matches/slightly beats it on the finite B1/B4 performance cases.

B32 exact-copy baseline before the accepted scheduler adapter was 128.258 us.
The current direct-port adapter measures 118.817 us, a 1.075x speedup while
remaining finite.

Current B32 IKP/nsys summary:

| kernel | launches | mean us | total ms |
|---|---:|---:|---:|
| `flash_fwd_splitkv_mla_fp8_sparse_kernel` | 1002 | 46.438 | 46.531 |
| `flash_fwd_mla_combine_kernel` | 501 | 23.078 | 11.562 |
| `get_mla_metadata_kernel` | 1 | 184.416 | 0.184 |

The producer remains the dominant kernel. Combine is the secondary target.
Scheduler metadata should be cached/reused in production; the benchmark excludes
it from steady-state preallocated timing.

## Known Limitation

The copied FlashMLA producer is currently safe when scheduler metadata gives each
split one 64-token block. Multi-block producer splits can produce all-NaN output
for random mixed packed-FP4 payloads. Constant KV, zero-Q plus constant KV, and
uniform e2m1 nibble patterns are finite. The likely issue is mixed-payload
multi-block producer/dequant/TMA/raw layout or SV accumulation, not individual
e2m1 values.

The 4096 scheduler cap is a correctness boundary for this direct port. The next
optimization phase should preserve the parity gate, then either fix multi-block
mixed-payload correctness to reduce split/combine pressure or optimize the
one-block producer/combine policy with IKP/CuTe/CZS/cutest.

# FlashMLA NVFP4 Sparse MLA Decode op-trt Mapping

## Reference

The native op-trt NVFP4 sparse MLA decode path is a direct FlashMLA port with a
narrow op-trt storage/scheduler adapter. The current reference is
`ai-blaise/FlashMLA` branch `ai-blaise/nvfp4-kv-decode-cute` at commit
`6b6a6ccebab96ef0d9a973bee251034122b63d86`.

The porting rule is strict: copy FlashMLA kernel/support source as-is where
possible, normalize only the op-trt boundary differences, then measure any
optimization against the prior op-trt port and the current FlashMLA behavior.

The source parity gate is:

```bash
python3 benchmarks/python/check_flashmla_nvfp4_source_parity.py   --flashmla-csrc /path/to/FlashMLA/csrc   --optrt-nvfp4-sparse cpp/tensorrt_llm/kernels/flashMLA/nvfp4_sparse
```

Current parity result against the latest reference:

| gate | result |
|---|---|
| strict exact copied files | 23 |
| config/kernel allowed delta | op-trt split data/scale pools instead of FlashMLA inline 336 B rows |
| scale-load allowed delta | scalar 32-bit loads because 36 B split-scale rows are not 16 B aligned |
| combine allowed delta | dispatch buckets through 1024 splits; zero dynamic shared-memory launch |

The copied surface targets only DeepSeek-V3.2 sparse MLA NVFP4 decode on B200.
op-trt intentionally omits unrelated FlashMLA API bindings, vendored CUTLASS,
head128/model1, BF16 decode, prefill, sm90, and q-prequant surfaces.

## op-trt Mapping

`cpp/tensorrt_llm/kernels/flashMLA/sparse_mla_decode_nvfp4.{h,cu}` maps op-trt
tensors/strides to FlashMLA `SparseAttnDecodeParams`.

`cpp/tensorrt_llm/thop/SparseMlaDecodeNvfp4Op.cpp` exposes:

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
| `kv` | `[num_pages, 64, 1, 288]`, packed NVFP4 score bytes |
| `kv_scales` | `[num_pages, 64, 1, 36]`, E4M3 scale bytes |
| `indices` | `[B, 1, topk]`, int32 token ids |
| `d_v` | 512 |
| page size | 64 |
| target topk | 1024 |

There is no FP8/BF16 KV fallback in this path. The current production boundary
uses op-trt's split NVFP4 data pool plus split E4M3 scale pool. FlashMLA's
latest inline 336 B row is the reference implementation and future storage
candidate, but request-time repacking is not used.

## Accepted Changes

1. Ported the latest FlashMLA direct PTX 9.2 conversion path for V3.2 K dequant:
   `cvt.rn.bf16x2.e2m1x2` and `cvt.rn.bf16x2.e4m3x2` replace the older
   f16x2 round trip.
2. Kept scalar 32-bit scale-row loads for op-trt split scales. The FlashMLA
   `uint4` scale load requires 16 B aligned inline scale rows; op-trt split rows
   are 36 B and can be misaligned.
3. Mirrored FlashMLA's zero dynamic shared-memory combine launch while keeping
   op-trt's added 256/512/1024 split buckets.
4. Updated scheduler policy from fixed `+15` overhead to a piecewise policy:
   `+15` for B8/B16/B32 and `+8` for B64/B128.

The direct BF16x2 PTX path requires the CUDA 13.2 ptxas/PTX 9.2 build path used
in the B200 buildtools environment. The verified container used the documented
`/tmp/ptxas_132_wrapper.sh` wrapper.

## Scheduler Policy

FlashMLA's default API uses `max(num_sms / s_q, 1)`. For random mixed packed-FP4
payloads under the V3.2 shape, that groups multiple 64-token top-k blocks into a
producer split and produces all-NaN output at B>=8.

op-trt computes scheduler metadata itself and uses the supplied metadata shape as
`num_sm_parts`. Current policy for topk=1024:

| B range | scheduler overhead | rationale |
|---|---:|---|
| B8/B16/B32 | `topk_blocks + 15` | fastest finite/equivalent point in sweeps |
| B64/B128 | `topk_blocks + 8` | same output as `+15`, lower producer/combine pressure |

The multi-block producer path remains a known correctness target. Until fixed,
this scheduler policy is part of the production correctness boundary.

## Verification

Focused gates were run on `a4-us-002-rl9` in the B200 buildtools image with the
narrow sparse-kernel/torch-extension harness.

Static/build gates:

| gate | result |
|---|---|
| source parity vs FlashMLA `6b6a6cc` with normalized op-trt deltas | pass |
| op-trt CMake target `flash_mla_nvfp4_sparse_src` | pass |
| ptxas for V3.2 producer | 168 registers, 16 barriers, 0 stack, 0 spills |
| B1 reference correctness topk64/topk1024 | pass |

Final B200 preallocated benchmark, `topk=1024`:

| B | scheduler parts | split count | finite | median us |
|---:|---:|---:|---|---:|
| 1 | 148 | 16 | yes | 30.796 |
| 4 | 148 | 64 | yes | 30.813 |
| 8 | 248 | 128 | yes | 30.842 |
| 16 | 496 | 256 | yes | 56.904 |
| 32 | 992 | 512 | yes | 114.765 |
| 64 | 1536 | 1024 | yes | 219.336 |
| 128 | 3072 | 2048 | yes | 430.147 |

Comparison to the previous op-trt direct-port adapter:

| B | previous median us | current median us | speedup |
|---:|---:|---:|---:|
| 1 | 30.790 | 30.796 | 0.999x |
| 4 | 31.221 | 30.813 | 1.013x |
| 8 | 32.876 | 30.842 | 1.066x |
| 16 | 57.533 | 56.904 | 1.027x |
| 32 | 118.817 | 114.765 | 1.035x |
| 64 | 236.524 | 219.336 | 1.076x |
| 128 | 466.570 | 430.147 | 1.077x |

Scheduler sweeps verified output equivalence against the previous safe policy:

| B | rejected lower parts | accepted best parts | previous parts | result |
|---:|---|---:|---:|---|
| 8 | 148, 160 non-finite | 248 | 248 | unchanged |
| 16 | 248, 320 non-finite | 496 | 496 | unchanged |
| 32 | <=640 non-finite; 768/896 finite but slower | 992 | 992 | unchanged |
| 64 | 1024/1280 non-finite | 1536 | 1984 | faster |
| 128 | 2048/2560 non-finite | 3072 | 3968 | faster |

Nsight Systems B32 attribution after the accepted changes:

| kernel | instances | avg ns | total ns | share |
|---|---:|---:|---:|---:|
| FlashMLA V3.2 producer | 214 | 43,518 | 9,312,926 | 76.1% |
| FlashMLA combine | 106 | 22,710 | 2,407,306 | 19.7% |
| scheduler metadata | 2 | 106,289 | 212,578 | 1.7% |

IKP CUPTI SASS collection selected producer/combine kernels but returned zero
`smsp__sass_inst_executed` in this container run. The raw artifact is retained,
but Nsight Systems is the usable profiler signal for this checkpoint.

## Rejected Candidates

| candidate | result | reason |
|---|---|---|
| FlashMLA `uint4` vector scale load on split scale pool | rejected | 36 B scale rows are not 16 B aligned; produced misaligned-address failure |
| lower B8/B16/B32 scheduler parts | rejected | non-finite or slower despite equivalent output |
| higher B32/B64/B128 scheduler parts | rejected | output-equivalent but slower |

## Remaining Work

This is still a BF16 dequant bridge around a pure NVFP4 cache. The next major
performance target is native full-NVFP4 tensor-core QK/PV integration using the
validated CuTe/CZS cache-row scaffold, while preserving the op-trt scheduler and
KV-manager contract. The FlashMLA inline 336 B row remains a candidate for
indigenization if it beats split-pool storage end to end, but not via request-time
repacking.

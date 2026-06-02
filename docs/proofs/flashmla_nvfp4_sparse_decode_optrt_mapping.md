# FlashMLA NVFP4 Sparse MLA Decode op-trt Mapping

## Reference

The native op-trt NVFP4 sparse MLA decode path is a direct FlashMLA port with a
narrow op-trt storage/scheduler adapter. The current source reference is
`ai-blaise/FlashMLA` branch `ai-blaise/nvfp4-kv-decode-cute` at commit
`6b6a6ccebab96ef0d9a973bee251034122b63d86` (`iter20+21: Q TMA
EVICT_LAST + L2_256B promotion for Q & K rope tensormaps`).

The porting rule is strict: copy FlashMLA kernel/support source as-is where
possible, normalize only the op-trt boundary differences, then measure every
optimization against the previous op-trt port and the executable FlashMLA
reference behavior when the FlashMLA extension is available.

The source parity gate is:

```bash
python3 benchmarks/python/check_flashmla_nvfp4_source_parity.py   --flashmla-csrc /path/to/FlashMLA/csrc   --optrt-nvfp4-sparse cpp/tensorrt_llm/kernels/flashMLA/nvfp4_sparse
```

Current parity result against `ai-blaise/nvfp4-kv-decode-cute` `6b6a6cc`:

| gate | result |
|---|---|
| strict exact copied files | 23 |
| config/kernel allowed delta | op-trt split data/scale pools instead of FlashMLA inline 336 B rows |
| conversion allowed delta | op-trt keeps direct `bf16x2` PTX conversion when the reference branch carries the older `f16x2` round trip |
| scale-load allowed delta | scalar 32-bit loads because 36 B split-scale rows are not 16 B aligned |
| combine allowed delta | dispatch buckets through 1024 splits; zero dynamic shared-memory launch |

The copied surface targets DeepSeek-V3.2 sparse MLA NVFP4 decode on B200. op-trt
intentionally omits unrelated FlashMLA API bindings, vendored CUTLASS,
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
uses op-trt's split NVFP4 data pool plus split E4M3 scale pool. FlashMLA's inline
336 B row remains a future storage candidate, but request-time repacking is not
used.

## Accepted Changes

1. Ported FlashMLA direct PTX 9.2 conversion for V3.2 K dequant:
   `cvt.rn.bf16x2.e2m1x2` and `cvt.rn.bf16x2.e4m3x2` replace the older
   f16x2 round trip.
2. Kept scalar 32-bit scale-row loads for op-trt split scales. The FlashMLA
   `uint4` scale load requires 16 B aligned inline scale rows; op-trt split rows
   are 36 B and can be misaligned.
3. Mirrored FlashMLA's zero dynamic shared-memory combine launch while keeping
   op-trt's added 256/512/1024 split buckets.
4. Updated scheduler policy from the previous fixed/long `+15/+8` policy to the
   smallest output-equivalent one-block scheduler verified by B200 sweeps:
   B1 uses `topk_blocks + 16`, B8/B16 use `+15`, B32 uses `+14`, and B64/B128
   use `+5`. For `topk < 1024`, the FlashMLA SM floor remains until separately
   swept.
5. Mirrored the iter20+21 cache hints that are valid for the op-trt split-pool
   mapping: Q TMA `EVICT_LAST` and Q/K RoPE tensormap `L2_256B` promotion.

The direct BF16x2 PTX path requires the CUDA 13.2 ptxas/PTX 9.2 build path used
in the B200 buildtools environment. The verified container used the documented
`/tmp/ptxas_132_wrapper.sh` wrapper.

## Scheduler Policy

FlashMLA's default API uses `max(num_sms / s_q, 1)`. For V3.2 NVFP4 at
`topk=1024`, B200 sweeps show that SM-count flooring over-splits B1 and excess
long-batch parts add combine pressure without improving correctness. The op-trt
adapter computes scheduler metadata itself and uses the supplied metadata shape
as `num_sm_parts`.

Current policy for `topk=1024`:

| B | scheduler overhead | scheduler parts | split count | status |
|---:|---:|---:|---:|---|
| 1 | 16 | 32 | 16 | accepted, faster than SM-floor 148 |
| 4 | 15 | 124 | 64 | accepted, equivalent to 148 |
| 8 | 15 | 248 | 128 | unchanged accepted point |
| 16 | 15 | 496 | 256 | unchanged accepted point |
| 32 | 14 | 960 | 512 | accepted vs 992 in same-seed sweep |
| 64 | 5 | 1344 | 1024 | accepted vs 1536 |
| 128 | 5 | 2688 | 2048 | accepted vs 3072 |

The multi-block producer path remains a correctness target. Until fixed, this
scheduler policy is part of the production correctness boundary.


## Executable FlashMLA Comparison

A FlashMLA reference extension was built on `a4-us-002-rl9` from
`ai-blaise/FlashMLA` `6b6a6cc` using CUTLASS `147f5673` and the B200
buildtools image. The CUDA 13.1 container path hit the PTX 9.2 `cvt`
assembler issue, so the successful build used the documented CUDA 13.2 ptxas
wrapper (`/tmp/ptxas_132_wrapper.sh`).

The executable comparison separates source parity, storage layout, and scheduler
safety:

| case | result | interpretation |
|---|---|---|
| source parity vs FlashMLA iter20+21 | pass with normalized op-trt deltas | op-trt keeps a direct-port kernel surface plus split-pool scheduler/combine boundary |
| B1/B4, FlashMLA compact scheduler vs op-trt split native | exact output match; op-trt is 1.059x faster at B1 and effectively equal at B4 | direct-port behavior matches the finite FlashMLA shapes and preserves iter20+21 cache hints |
| B8+ at `topk=1024`, FlashMLA compact scheduler | FlashMLA output/lse becomes non-finite; op-trt split native remains finite | FlashMLA's compact scheduler is not a production-correct baseline for the target batch/topk range |
| FlashMLA compact scheduler metadata reused in op-trt | reproduces non-finite B8+ behavior | the failure follows scheduler/split coverage, not split-vs-inline storage |
| exact inline 336 B row scratch with op-trt safe scheduler | exact at B1/B4, finite at B8+, but slower than split storage at every measured B | inline storage is a reference layout, not a promotion candidate for op-trt today |

Therefore, the required production delta is the op-trt scheduler/combine adapter
layered over the direct FlashMLA producer, while preserving split NVFP4 data and
E4M3 scale pools. The iter20+21 FlashMLA branch remains the layout/cache-hint
reference, but its compact scheduler timings are not accepted when the output is
non-finite.

Latest native-only B200 gate, `topk=1024`, cached focused extension:

| B | scheduler parts | split count | finite | median us |
|---:|---:|---:|---|---:|
| 1 | 32 | 16 | yes | 29.636 |
| 4 | 124 | 64 | yes | 30.815 |
| 8 | 248 | 128 | yes | 30.922 |
| 16 | 496 | 256 | yes | 57.458 |
| 32 | 960 | 512 | yes | 114.799 |
| 64 | 1344 | 1024 | yes | 214.270 |
| 128 | 2688 | 2048 | yes | 422.124 |


## Verification

Focused gates were run on `a4-us-002-rl9` in the B200 buildtools image with the
narrow sparse-kernel/torch-extension harness.

Static/build gates:

| gate | result |
|---|---|
| source parity vs FlashMLA `6b6a6cc` with normalized op-trt deltas | pass |
| `git diff --check` | pass |
| `py_compile` for touched Python benchmark/parity files | pass |
| focused torch-extension rebuild | pass |
| ptxas for V3.2 producer | 168 registers, 16 barriers, 0 stack, 0 spills |
| B1 reference correctness topk64/topk1024 | pass |

Final B200 preallocated native-only benchmark after the raw NoPE TMA
`EVICT_LAST` promotion, `topk=1024`:

| B | scheduler parts | split count | finite | median us |
|---:|---:|---:|---|---:|
| 1 | 32 | 16 | yes | 29.388 |
| 4 | 124 | 64 | yes | 30.784 |
| 8 | 248 | 128 | yes | 30.831 |
| 16 | 496 | 256 | yes | 56.088 |
| 32 | 960 | 512 | yes | 111.688 |
| 64 | 1344 | 1024 | yes | 214.272 |
| 128 | 2688 | 2048 | yes | 421.634 |

Final gate artifact: `.logs/raw_tma_evict_last_final_20260602T165351.json`.

Raw NoPE TMA cache-policy A/B, same focused harness settings
(`warmup=80`, `iters=220`, native-only, `topk=1024`):

| B | `EVICT_NORMAL` median us | `EVICT_LAST` median us | speedup |
|---:|---:|---:|---:|
| 1 | 29.267 | 29.144 | 1.004x |
| 4 | 30.783 | 30.788 | 1.000x |
| 8 | 30.818 | 30.816 | 1.000x |
| 16 | 56.960 | 56.008 | 1.017x |
| 32 | 114.733 | 111.657 | 1.028x |
| 64 | 214.538 | 214.238 | 1.001x |
| 128 | 421.928 | 421.654 | 1.001x |

`EVICT_FIRST` was also functionally valid, but `EVICT_LAST` was better at B1,
B16, B32, and was neutral elsewhere. The promoted change is only the raw NoPE
TMA gather hint; Q TMA and K/RoPE tensormap promotions remain the iter20+21
FlashMLA settings.

Same-seed scheduler confirmation for changed points:

| B | previous parts | accepted parts | previous median us | accepted median us | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 148 | 32 | 30.784 | 29.671 | 1.038x |
| 32 | 992 | 960 | 114.772 | 112.526 | 1.020x |
| 64 | 1536 | 1344 | 219.398 | 215.115 | 1.020x |
| 128 | 3072 | 2688 | 430.966 | 422.699 | 1.020x |

Comparison to the previous pushed op-trt direct-port adapter (`c462c4a69`) using
its recorded final matrix:

| B | previous median us | current median us | speedup |
|---:|---:|---:|---:|
| 1 | 30.796 | 29.783 | 1.034x |
| 4 | 30.813 | 30.818 | 1.000x |
| 8 | 30.842 | 30.944 | 0.997x |
| 16 | 56.904 | 57.432 | 0.991x |
| 32 | 114.765 | 114.767 | 1.000x |
| 64 | 219.336 | 214.384 | 1.023x |
| 128 | 430.147 | 422.243 | 1.019x |

B8/B16 logic is unchanged; the small differences above are run-to-run variance
from independent final matrices, not a scheduler policy change.

Nsight Systems B32 attribution from the previous checkpoint remains the usable
profiler signal: producer ~76%, combine ~20%, scheduler metadata ~2%. IKP CUPTI
SASS collection selected producer/combine kernels but returned zero
`smsp__sass_inst_executed` in the container run, so Nsight Systems remains the
reliable profiler for this checkpoint.

## Rejected Candidates

| candidate | result | reason |
|---|---|---|
| raw NoPE TMA `EVICT_FIRST` | rejected | correct and slightly faster than normal at B16/B32, but slower than `EVICT_LAST` at the main changed points |
| raw NoPE TMA `EVICT_NORMAL` | replaced | same-run A/B showed `EVICT_LAST` wins at B1/B16/B32 and is neutral elsewhere |
| FlashMLA `uint4` vector scale load on split scale pool | rejected | 36 B scale rows are not 16 B aligned; produced misaligned-address failure |
| FlashMLA inline 336 B row in op-trt safe scheduler | rejected | exact/finite but slower than split storage: B1 38.99 us, B4 57.45 us, B8 80.46 us, B16 151.68 us, B32 294.47 us, B64 473.81 us, B128 937.94 us |
| FlashMLA iter20+21 compact scheduler as production baseline | rejected | non-finite for `topk=1024` at B8 and above |
| B4 64 parts | rejected | non-finite, split count collapsed to 32 |
| B8 160 parts | rejected | non-finite, split count collapsed to 64 |
| B16 320 parts | rejected | non-finite, split count collapsed to 128 |
| B32 <=640 parts | rejected | non-finite or incomplete split coverage |
| B64 1280 parts | rejected | non-finite, split count collapsed to 512 |
| B128 2560 parts | rejected | non-finite, split count collapsed to 1024 |
| higher long-batch parts | rejected | output-equivalent but slower from extra producer/combine pressure |

## Remaining Work

This is still a BF16 dequant bridge around a pure NVFP4 cache. The next major
performance target is native full-NVFP4 tensor-core QK/PV integration using the
validated CuTe/CZS cache-row scaffold, while preserving the op-trt scheduler and
KV-manager contract. FlashMLA remains the source/layout reference; production
acceptance requires finite outputs across the target B/topk range, so op-trt
should not add request-time repacking to chase the inline 336 B layout.

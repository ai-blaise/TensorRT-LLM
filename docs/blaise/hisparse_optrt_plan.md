# OP-TRT HiSparse Integration Plan

This document is the second-pass integration plan for building a HiSparse-style
hierarchical sparse-attention KV path on top of the current `op-trt` custom
stack. It is based on direct review of:

- LMSYS/SGLang HiSparse blog:
  https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/
- SGLang HiSparse guide:
  https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hisparse_guide.md
- SGLang implementation, re-checked against `sgl-project/sglang` main
  `f7041c9dee2263824a128ef4941448e41c500789` on June 13, 2026:
  - `python/sglang/srt/managers/hisparse_coordinator.py`
  - `python/sglang/srt/mem_cache/allocator/hisparse.py`
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
  - `python/sglang/jit_kernel/csrc/hisparse.cuh`
  - `python/sglang/jit_kernel/hisparse.py`
  - `python/sglang/srt/layers/attention/dsa_backend.py`
  - `python/sglang/srt/layers/attention/dsv4/indexer.py`
  - `python/sglang/srt/disaggregation/decode.py`
  - `sgl-kernel/python/sgl_kernel/top_k.py`
- OP-TRT implementation:
  - `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  - `tensorrt_llm/_torch/attention_backend/sparse/kvarn_backend.py`
  - `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
  - `tensorrt_llm/_torch/disaggregation/transceiver.py`
  - `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`
  - `tensorrt_llm/_torch/disaggregation/resource/utils.py`
- Dynamo/NIXL write-mode docs:
  - `docs/api/nixl-connect/writable-operation.md`
  - `docs/api/nixl-connect/write-operation.md`

## Source-Of-Truth Correction

Some older AgentMemory notes use `HISA` and `HiSparse` interchangeably. Do not
use those notes as implementation targets. In this branch, the current
`op-trt` code is the source of truth for serving semantics:

- `HISA` is the existing Indexer-side candidate scoring/index-cache path and
  keeps its current FP4/device-resident contracts;
- `HiSparse` is the new hierarchical host/hot KV transfer and sparse-MLA
  consumption path being added around that existing scoring output;
- CuTe, CZS, IKP, CUTLASS, `cutest`, and related AgentMemory materials are
  only optimization/proof-process inputs: use them to decide how to compile,
  prove, profile, and promote kernels, not to replace the branch's production
  ABI or dataflow;
- any optimization candidate that changes request identity, row-status,
  KVarN BDR layout, Indexer/HISA output semantics, NIXL write-mode ordering, or
  LayerSplit ownership must be rejected unless the current `op-trt` production
  path and tests are updated explicitly and proved end to end.

## CuTe, CZS, And IKP Optimization Gate

The June 13 audit re-read the relevant AgentMemory entries directly from the
canonical memory file behind port `3811`. Only the CuTe/CZS/IKP process
requirements are carried into this plan:

- CZS is the required compiler/proof path for CuTe candidates. The active
  memory entry points to `ai-blaise/CZS` at
  `148ed9fadc886617f1473249994a4279170eb98e` with `czs prove --json FILE`,
  `czs walk FILE`, JSON round-trip coverage for Layout, Swizzle,
  SwizzledLayout, TMA, `ldmatrix`, vectorization, mbarrier, Tensor Memory,
  distributed shared memory, barrier elision, MMA overlap, and CLC sites, plus
  recorded build/ctest/pytest/thread-pool validation.
- IKP evidence is required before promoting any kernel optimization beyond
  correctness/proof smoke. A promoted candidate must archive source, benchmark
  JSON, ptxas register/spill output, occupancy/shared-memory notes, IKP trace
  summaries, and NCU or CUPTI stall evidence for both accepted and rejected
  variants.
- Blackwell/CuTe material should guide layout only when it matches the
  production data layout. The current KVarN-hot reader keeps packed global
  `kvarn_k2v2` BDR records and byte-addressed scale/zp fields; a CuTe/UMMA
  candidate cannot assume NVF4 hardware block-scale layout unless it introduces
  and proves an explicit conversion that preserves the production ABI.
- The current direct `sparse_mla_decode_kvarn_hot` kernel is acceptable as a
  fail-closed production-ABI proof path. The next performance candidate should
  first import only the compatible FlashMLA split-scheduler/combine structure
  while preserving KVarN BDR producer loads and resident sink/tail reads. Any
  later CuTe rewrite must pass the same serving-layout import proof and add CZS
  proof plus IKP artifacts before it can replace the direct path.
- Whole-system optimization remains higher priority than isolated instruction
  polish until serving proof closes: request-table lifecycle batching,
  NIXL write-mode ordering, host/hot admission, compact miss scheduling,
  row-status fail-closed behavior, and sparse-MLA hot-index construction are
  part of the same performance target as the CUDA kernel.
- On `a4-us-001`, the available proof/profiling tools are
  `/home/spencer/work/CZS/build/src/czs` (`czs 0.4.1`),
  `/home/spencer/work/intra-kernel-profiler` (use
  `docs/integration_existing_kernels.md` and `docs/trace_tips.md` for region
  placement/capacity), and `/home/spencer/work/cutest` for simple CuTe DSL
  fusion experiments. CZS source there is an exported source tree without
  `.git`; the authoritative CZS commit remains the AgentMemory/GitHub
  `148ed9fadc886617f1473249994a4279170eb98e` record above.
- The June 13 build-wait audit re-read the local CZS and IKP documents on
  `a4-us-001` directly, not through summaries. The HiSparse promotion checklist
  now treats the following CZS obligations as concrete gates for any replacement
  of the direct KVarN-hot sparse MLA kernel:
  - `v4_sub_byte_alignment`: prove sub-byte packed runs with 16-byte base
    alignment and leading dimensions that satisfy the packed operand contract;
  - `v3_mma_overlap`: prove any shared-SMEM reuse between future-stage
    operand regions and output regions before reducing pipeline separation;
  - `v3_clc_race_freeness`: prove each CTA in a CLC cluster issues
    `try_cancel` exactly once before adopting persistent dynamic scheduling;
  - `v4_tcgen_cp_scale_staging`: prove scale-factor staging only for kernels
    that actually use Blackwell block-scale MMA layouts. The current
    `kvarn_k2v2` BDR record is byte-addressed KVarN storage, not NVFP4
    hardware block-scale storage, so this proof cannot be used to justify a
    layout substitution.
- IKP integration is similarly gate-level, not optional polish. The first
  instrumented profile must keep the region set small enough to avoid trace
  perturbation and must cover at least: full kernel, TopK/hot-index decode,
  BDR read/dequant, resident sink/tail read, score reduction/softmax, value
  accumulation, and output write. The accepted optimization package must also
  include an NSys merge so launch latency, NIXL/host-hot copies, CUDA API gaps,
  and NCCL/LayerSplit traffic are visible on the same timeline as the
  intra-kernel regions.
- The active direct CUDA kernel has already compiled in the full `th_common`
  build with ptxas reporting no spills for
  `sparse_mla_decode_kvarn_hot.cu` on SM100a, but that is only a compilation
  signal. It does not replace the CZS/IKP/NSys promotion evidence above, and it
  does not authorize a split-producer/CuTe rewrite before the serving-layout
  import proof and live DSA/NIXL proof are complete.

## Executive Decision

Do not copy SGLang HiSparse wholesale. Build an OP-TRT HiSparse subsystem that
uses SGLang's proven architecture:

1. full logical KV in host-pinned memory;
2. fixed-size hot KV buffer in GPU memory;
3. top-k-driven swap-in with hit/miss/LRU in CUDA;
4. direct-to-host PD transfer through write-mode metadata;
5. eager backup of newly produced decode KV.

But adapt it around OP-TRT's custom contracts:

1. Indexer/HISA scoring remains the source of truth;
2. Indexer K remains resident `fp4` and is not KVarN;
3. dense MLA latent KV remains production `kvarn_k2v2`;
4. host/hot ownership must be block-oriented, because KVarN restore,
   LayerSplit broadcasts, NIXL page tables, and sparse MLA all key on paged
   block ids;
5. LayerSplit owner-local prefill and TP4/CP1 decode must remain valid;
6. SMC-SD/Moondream decode must not reuse host slots before speculative cleanup;
7. no silent fallback to non-custom paths is allowed.

The full implementation path must be based on the target model's production
architecture from the first runtime-reachable HiSparse serving candidate. That
means dense MLA `kvarn_k2v2` cold/hot storage, FP4 Indexer K + HISA scoring,
sparse MLA with BDR/on-read dequant, NIXL generation-first direct-to-host, and
the r20 LayerSplit/SMC/Moondream wiring. Do not implement FP16 host/hot tiers
in serving code. Independent test fixtures may compare against reference
tensors outside the HiSparse coordinator/transceiver path, but there is no
FP16 block-hot oracle in the serving implementation, runtime fallback, config
mode, or deployment candidate. A path is considered runtime-reachable if it can
be selected by a manifest, config flag, coordinator branch, transceiver branch,
attention dispatch, or kernel ABI used by a serving request.

The startup/mapping guard is intentionally granular: enabled HiSparse first
requires the native planner/copy ops, then the production BDR hot-reader
primitive, then a fused `sparse_mla_decode_kvarn_hot` sparse MLA dispatch, then
the explicit sink/tail resident-token readiness probe. The direct fused kernel
now consumes resident normal-KV sink/tail tokens through the production
descriptor ABI, but the readiness probe intentionally returns false until that
path is rebuilt in the deployment image and live-proven with DSA row metadata,
NIXL admission, cleanup, and stale-row rejection. This avoids the earlier
stale guard wording without promoting an unproven runtime path.

Serving acceptance is binary: if `hisparse_enabled=true` can answer a request
before dense-MLA KVarN BDR source writes, typed NIXL direct-to-host, native
device-side hot planning, stream-ordered packed host-to-hot copy, post-copy hot
metadata commit, and sparse MLA KVarN-hot BDR/on-read dequant are all wired and
live-proven together, that is a correctness bug. A partial component may exist
only as a fail-closed production-ABI scaffold or as an offline test fixture
that is unreachable from the coordinator, transceiver, kernel ABI, deployment
config, and runtime fallback policy.

Current branch posture after the June 13 final thoroughness/correctness sweep:
the branch has a production-architecture HiSparse path with fail-closed
promotion gates, not a deployable HiSparse serving candidate. The config
validation, packed KVarN tier allocation, host metadata publication, NIXL DRAM
registration, request
host-slot sideband, packed KVarN source/destination fragment derivation, typed
`HISPARSE_HOST` write submission, decode admission state, and two-stage
hot-block planning ABI are implemented. A device-visible request table now
publishes disaggregated request ids, request-relative block-to-host-slot rows,
host commit generations, and admission flags for the future native hot-slot
planner. This table is the required production ABI shape, and request-table
lifecycle updates now publish through host-side rows followed by native,
stream-ordered row/slot copies to the device mirror instead of per-cell device
scalar writes. Enabled CUDA HiSparse now rejects request-table publication if
`trtllm::hisparse_publish_request_table_slots` is unavailable. A future
multi-slot batcher can still coalesce lifecycle events further, but the device
mirror is no longer updated through Python scalar writes. The sender now returns
explicit
`(local_layer, request_block_pos)` commit coverage only after the normal KV
write and typed host write both succeed, and the receiver accumulates that
coverage before marking host records committed. Admission is explicit: a
request cannot be marked HiSparse-ready unless all reserved prompt host blocks
are committed and no host writes are pending. Host-to-hot planning is also
explicit: the coordinator may plan packed KVarN miss copies, but it does not
publish hot residency until the planned native copy is accepted. A strict
native thop now exists for packed KVarN host-to-hot copies, and a native
CUDA-side TopK-to-block dedupe primitive now exists for the first planner stage.
A native request-table resolver now maps device block rows and row request ids
to committed host slots and commit generations, returning explicit invalid
status flags for missing, unadmitted, out-of-range, or uncommitted rows rather
than falling back to Python request-table extraction. A native non-mutating
hot-slot planner now consumes those resolved rows and layer-local hot metadata
to produce hit/miss/LRU slot decisions, copy schedules, and row status without
publishing residency before packed copies succeed. A native compact miss
schedule op now turns row-major device miss tensors into contiguous device
`(host_slot, hot_slot, row_id)` vectors plus a device copy count. A native
`trtllm::hisparse_submit_packed_kvarn_copy_schedule` bridge now consumes that
compact device schedule directly and copies from mapped pinned host KVarN
storage into hot HBM in stream order, returning per-row copy status for the
post-copy commit stage. It intentionally fails closed if the host tier is not
device-addressable; it does not use a CUDA host callback to enqueue copies and
does not synchronously read the schedule back to Python. A native post-copy hot
metadata commit op now publishes `hot_host_slot`, `hot_commit_gen`, and
`hot_lru_tick` on device only after the packed-copy stage has accepted the plan.
The branch also has a native hot-index builder that preserves the existing
`base * stride_factor + layer_idx * tokens_per_block + token_offset` sparse-MLA
index contract while targeting HiSparse hot slots instead of full-pool blocks.
The June 13 continuation corrected the native block-dedupe row budget so
resident sink/tail blocks are admitted before hot-slot planning: dedupe now
budgets hot capacity plus configured sink blocks plus one possible tail block,
while the planner still fails closed if committed-hot blocks exceed hot
capacity. This avoids prematurely rejecting valid sink/tail-heavy rows without
creating a hidden fallback or overcommitting hot HBM.
The same sweep also separated HiSparse's hot-slot attention index space from
LayerSplit's dense normal-KV broadcast index space. When HiSparse mapping is
active, `sparse_attn_predict()` still returns hot-pool indices to sparse MLA,
but the LayerSplit legacy dense read-set is recomputed from the original TopK
through the normal local-to-global KV transform before calling
`_layersplit_topk_global_block_ids()`. This prevents hot-slot ids from being
interpreted as global KV block ids. It is conservative and correct; after live
proof, the next optimization is to shrink that broadcast read-set to resident
sink/tail blocks only, because committed full blocks should be consumed from
the HiSparse hot tier rather than dense normal KV.
The coordinator now chains those native stages through hot-index construction
when real CUDA TopK/request metadata is present, then fails closed at the
remaining sparse MLA hot-pool read/BDR dequant gate.
The branch now also has a native production-layout BDR hot-reader primitive,
`trtllm::hisparse_read_kvarn_hot_bdr`, that validates the generated hot-index
encoding, consumes row status, decodes packed `kvarn_k2v2` hot records through
the C-KV low-bit bytes, scale/zp fields, and E4M3 RoPE payload, and returns a
bf16 scratch tensor for kernel validation. This is a producer-load building
block and CUDA smoke hook, not a serving pre-dequant path; the fused sparse MLA
candidate now uses the same reader helpers directly at producer load. The June
13 final sweep tightened this primitive to require `kvarn_bits=2`; there is no
4-bit KVarN-hot validation branch for the production HiSparse path.
The BDR address decode, hot-index validation, 2-bit C-KV unpack, scale/zp
application, inverse 128-wide BDR/Hadamard readback, and E4M3 RoPE byte read
now live in `hisparseKvarnBdrRead.cuh`. The scale/zp payload is treated as a
byte-addressed BDR record field rather than a half-aligned tensor field, so
odd byte strides in the packed slot layout cannot fault or silently corrupt
reads. The standalone hot-reader smoke op and fused sparse MLA kernel use that
same device helper layer so the validated CUDA smoke path and serving
producer-load path cannot drift.
The coordinator also now constructs a typed
`HiSparseSparseMlaKvarnHotDescriptor` at the native-chain boundary. That
descriptor carries `hot_packed`, hot global indices, row status, fixed-top-k
semantics, layer, stride, capacity, packed-record size, and the production
`kvarn_k2v2` dense-MLA dimensions. It is built only from real native outputs
and is returned to the absorption-generation dispatch path; it is not a serving
fallback and does not reconstruct loose hot-pool tensors.
The branch now has the first native `trtllm::sparse_mla_decode_kvarn_hot`
operator. It is not a wrapper over `sparse_mla_decode_nvfp4`: the CUDA kernel
reads packed `kvarn_k2v2` BDR hot records through `hisparseKvarnBdrRead.cuh`,
reconstructs dense-MLA C-KV values from the BDR/Hadamard-domain record on read,
computes scores against the 576-wide dense-MLA key, applies softmax, and emits
the 512-wide latent value output. It also consumes the `explicit_sink_tail_v1`
resident-read sentinel: committed blocks read packed-hot BDR records, while
sink/tail hits resolve the original request-relative TopK token through the
live normal decode block table and read bf16/fp16 latent K/V from the resident
normal-KV pool. Because the native resident classifier is block-major, the
descriptor and op wrapper now require resident `sink_tokens` to be an exact
multiple of `tokens_per_block`; partial sink blocks fail closed instead of
being truncated into `sink_tokens // tokens_per_block`. This is a real
KVarN-hot producer-load path, and the HiSparse
absorption-generation branch now calls it through the typed descriptor before
any NVFP4 or full-HBM path can run. It is not yet promoted: the new
`torch.ops.trtllm.hisparse_sparse_mla_resident_v1_ready()` readiness surface
currently returns false until live runtime proof and profiling are complete,
and the kernel still uses a direct per-row/head schedule plus scalar inverse
BDR readback before the optimized FlashMLA-style split scheduler/query-fold
implementation is imported. The June 13 final sweep tightened the fused
operator ABI guard so `hot_packed` records must be at least the production
`kvarn_k2v2` BDR byte size before launch; a too-short hot record now fails in
the C++ wrapper instead of allowing a CUDA out-of-record read.
The dispatch keys generation sparse-MLA shape on `num_generations`, not total
mixed-batch sequence count, and the coordinator slices generation request IDs
before resolving hot host slots. That keeps mixed prefill+decode batches from
binding generation rows to context request IDs. The absorption-generation
dispatch also treats cached sparse-MLA descriptors as layer-local: a descriptor
is reused only when the layer id, row count, row-status count, TopK width, and
CUDA devices match the current call. Otherwise it remaps through the
coordinator and still fails closed rather than consuming stale hot-slot state.
The kernel translation unit has been non-disruptively compiled on the B200 VM
with CUDA 13 (`nvcc -arch=sm_100`) without allocating GPU memory. The June 13
final sweep also added and proved a narrow exact-clean `th_hisparse_smoke`
target that links the real HiSparse torch registrations and production CUDA
kernels without pulling the unrelated generated CUTLASS/MoE tail of
`th_common`. That target was built in
`/home/spencer/work/TensorRT-LLM-hisparse-runtime` with the persistent
`/home/spencer/work/build-cache/hisparse-thop` cache, then loaded on B200 GPU 7
through `torch.ops.load_library`. A direct CUDA smoke using the production
constants passed for native BDR record write, byte-strided scale/zp layout,
hot BDR readback, sparse MLA KVarN-hot decode, and resident sink/tail padding.
The follow-up
`blaise_perf/hisparse/native_planner_copy_smoke.py` proof now exercises the
native planner/copy chain with real CUDA tensors: TopK-to-block dedupe,
resident sink/tail classification, request-table resolve, hot-slot planning,
compact miss scheduling, mapped pinned-host host-to-hot copy, post-copy
metadata commit, hot-index construction, second-pass hit reuse, overflow
fail-closed behavior, unadmitted-row rejection, and uncommitted-block
rejection. It passed on B200 GPU 7 both in the exact-clean buildtools image and
inside the newest local deployment runtime image while loading the mounted
`libth_hisparse_smoke.so`. The tiny
`deploy/disagg_pd_r20/build_hisparse_thop_proof_image.sh` proof image then
copied that exact branch-built library and script into the deployment runtime
base and reran the same CUDA smoke with no thop bind mount; image-internal
native-op loading also passed on B200 GPU 7. This is still narrower than a full
serving wheel: it proves runtime ABI/device-addressable host access for the
branch-built HiSparse thops, not DSA serving import from `libth_common.so`.
The same planner/copy smoke also passes when loading the cached branch-built
`libth_common.so` directly: first mounted into the buildtools image, then
mounted into the deployment runtime image, and finally copied into a tiny
deployment-runtime proof image together with its required native siblings
(`libtensorrt_llm.so`, `libpg_utils.so`, and the decoder-attention shared
libraries). That image-internal `libth_common.so` proof closes the native
library dependency gap that first appeared as a missing `libtensorrt_llm.so`;
the remaining packaging gap is installing the full branch Python package plus
`libth_common.so` into site-packages exactly as serving imports it.
`deploy/disagg_pd_r20/build_hisparse_serving_import_proof_image.sh` is the
next low-cost gate for that packaging boundary: it overlays the current branch
`tensorrt_llm` package into deployment-runtime site-packages, places the
branch-built `libth_common.so` under `tensorrt_llm/libs`, imports
`tensorrt_llm` normally, asserts that the normal package loader used that
serving-layout library, and then runs the native planner/copy smoke without an
explicit `--library` path. The June 13 follow-up sweep extended that helper so
it can also carry branch-built generated package-root artifacts
(`bindings*.so`, `tensorrt_llm_transfer_agent_binding*.so`) and package library
artifacts (`libnvinfer_plugin_tensorrt_llm.so`, transfer wrappers) from the
persistent VM build cache when those artifacts are present. This narrows the
remaining fullsource gap to proving those generated artifacts from the same
branch/cache inside the serving import path, then promoting only after live
DSA/NIXL deployment proof.
The committed serving-layout proof now passes on B200 GPU 7:
`localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-c6143d16492f-hisparse-serving-import-proof-20260613T130103Z`
was built from source SHA `c6143d16492f4dde16375eb28f68e43298e10d7e`, imported
`tensorrt_llm` from `/opt/dynamo/venv/lib/python3.12/site-packages`, loaded
`/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libth_common.so`
through the normal package loader, and completed
`native_planner_copy_smoke.py` without an explicit `--library` path.
The same serving-layout proof was rerun after the June 13 final plan/helper
sweep at source SHA `aba720ca642256ec710daf1bd4e2a2409cb1e8a6`:
`localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-aba720ca6422-hisparse-serving-import-proof-v2-20260613T131600Z`
overlays branch Python into deployment-runtime site-packages, loads
`site-packages/tensorrt_llm/libs/libth_common.so` through normal import, and
passes `native_planner_copy_smoke.py` without `--library`. That proof image
also carried the cached branch-built NIXL and UCX wrapper libraries alongside
`libtensorrt_llm.so`, `libpg_utils.so`, and decoder-attention native siblings.
`package_root_files` and `package_lib_files` were empty because a current diff
audit against `origin/op-trt` found no branch changes under
`cpp/tensorrt_llm/nanobind`, `cpp/tensorrt_llm/plugins`, or the transfer-agent
binding source; the branch-generated Python binding/plugin artifacts are
therefore not a current-head correctness delta. If those sources change later,
the generated-artifact proof remains mandatory before promotion.
The repo-level pytest harness still requires the full Python bindings, so the
proof script intentionally bypassed `tests/unittest/conftest.py` while
executing the same native ops and tensor contracts.
The June 13 final correctness sweep strengthened the serving-layout proof so
`serving_import_smoke.py` now runs both the native planner/copy chain and the
fused sparse MLA KVarN-hot CUDA smoke through the normal package import path.
The added `blaise_perf/hisparse/sparse_mla_kvarn_hot_smoke.py` directly
exercises `trtllm::mla_bdr_write_kvarn_record` plus
`trtllm::sparse_mla_decode_kvarn_hot` over production `kvarn_k2v2` constants,
including an odd packed-record stride to prove byte-addressed scale/zp fields
and an explicit sink/tail resident-token row, all-padding row, and invalid
resident-request-id row. This is a production-layout native smoke, not an FP16
block-hot oracle and not a serving fallback. The
exact-clean `libth_hisparse_smoke.so` still passes that stronger smoke on B200
GPU 7. The same strengthened script was rerun on `a4-us-001` against the
resident narrow `libth_hisparse_smoke.so` in
`/home/spencer/work/build-cache/hisparse-splitprod/cpp-build/tensorrt_llm/thop`
and passed committed-hot, resident sink/tail, all-padding, invalid resident
request id, stale layer, and stale hot-slot cases. This is still a narrow
native-op proof, not the serving-layout `libth_common.so` proof. The older
cached package `libth_common.so` failed the stronger smoke
with a CUDA misaligned-address error, which is now treated as a stale-library
finding rather than a source-level BDR-layout failure because the exact-clean
library uses the current byte-addressed BDR helpers successfully. A rebuilt
`libth_common.so` plus a rerun of
`deploy/disagg_pd_r20/build_hisparse_serving_import_proof_image.sh` is required
before the stronger serving-import proof can be marked complete.
The rebuilt `libth_common.so` serving-layout proof is now complete on
`a4-us-001` for source SHA `4d274bd53338726648d9fa6c05e0d5b70375a3df`:
`localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-4d274bd53338-hisparse-serving-import-proof-20260613T142136Z`
was built from the persistent
`/home/spencer/work/build-cache/hisparse-thop-001` cache, imported branch
Python from `/opt/dynamo/venv/lib/python3.12/site-packages`, loaded
`/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libth_common.so`
through the normal package loader, and passed both
`native_planner_copy_smoke.py` and `sparse_mla_kvarn_hot_smoke.py` without an
explicit `--library` path. This closes the stale-`libth_common.so` serving
import gap. The remaining gate is live DSA/NIXL deployment proof and
profiling; this image proof still does not flip
`hisparse_sparse_mla_resident_v1_ready()` to true. All follow-up VM work for
this branch should stay on `a4-us-001`; `a4-us-002` is not part of the current
allowed workflow. The same audit confirmed the live Dynamo deployment is
running the NIXL/PYTHON transceiver configuration with `--connector none`; the
explicit
`TRTLLM_NIXL_KVCACHE_BACKEND=UCX` value is the NIXL plugin backend on the
current GCP B200 lane, not the legacy/direct UCX cache transceiver.
The June 13 sweep also fixed a separate dense-MLA KVarN correctness issue in
`mlaKernels.cu`: the paged MLA KVarN read launcher validated `kvarn_bits` but
did not pass it into the CUDA kernel, which meant `kvarn_k2v2` could be read
through the default 4-bit path. The launcher now forwards the validated bit
width, so dense MLA KVarN readback uses the selected 2-bit production mode.
The coordinator refuses to derive HiSparse host/hot tier sizes from the legacy
Python/Sinkhorn `KVarNLatentPool` record. HiSparse tier derivation requires a
production BDR layout descriptor (`bdr_ckv_lowbit_fp8_pe_v1`) and a matching
source-pool layout. When `hisparse_enabled=true`, DSA now allocates a separate
production-shaped BDR source pool per local layer and marks that as the
HiSparse source layout; the legacy side-pool remains only for amortized
restore. The direct-to-host fragment API has the same guard and will not
publish legacy side-pool pointers as HiSparse host-write sources.
The branch now also registers `torch.ops.trtllm.mla_bdr_write_kvarn_record` and
calls it from the full-block KVarN commit walk when the HiSparse BDR source pool
is active. That native writer consumes the production paged latent block view
and fills C-KV low-bit bytes, C-KV scale/zp bytes, and the current 8-bit RoPE
payload in `KVarNBDRSourcePool`. The writer now emits C-KV scale/zp as bytes
using CUDA's public half raw-conversion intrinsics, so a packed record remains
valid even when the slot stride is not half-aligned. Blocks that were already
committed to the legacy side-pool are backfilled into the BDR source pool
instead of being skipped.
The dense MLA decode branch also fails closed when a HiSparse coordinator is
enabled, so an accidentally relaxed planner guard cannot route KVarN-hot
indices through `sparse_mla_decode_nvfp4` or the restored full-pool TRTLLM MLA
path.
Startup and runtime mapping still intentionally reject `hisparse_enabled=true`
before serving because the resident-v1 sparse MLA readiness probe remains
false until live DSA/NIXL/B200 deployment proof is complete. The sparse MLA
hot-pool read, BDR/on-read dequant, native BDR writer, and resident sink/tail
producer-load paths are now present and smoke-proven at the native op level,
but promotion still requires live DSA row-status proof, writer stream ordering
against NIXL source reads, live NIXL/cancel E2E proof, CUDA graph lifecycle
proof, and profiling. This is the correct failure mode: no manifest should get
an implicit full-HBM, FP16-staging, Python TopK extraction, or
direct-to-host-off substitute.
The June 13 cancellation audit also hardened the receiver side of that live
NIXL gate: if an `RxSession` is already `ERROR` or `CANCELLED`, a late
`KV_AGENT_RESULT SUCCESS` from a transferring task now finishes the pending
HiSparse host write and fails the task without recording commit coverage,
marking host blocks valid, or admitting the request. This prevents the
transient "cancelled but admitted" state before the eventual force-release
cleanup. The behavior was verified in the deployment-runtime proof image by
overlaying the patched `transfer.py` onto the installed package and running a
synthetic cancelled-session host-write scenario; it printed
`late-success cancel guard passed`.
The same audit then closed the dispatch-window race where a generation-side
cancel can arrive after HiSparse host slots are reserved but before
`Receiver.dispatch_task()` marks the receive task as transferring. INIT tasks
now finish their pending HiSparse host-write counter during cancel, sender
endpoints are captured before the transfer state transition, and
`mark_transferring()` returns false instead of resurrecting a terminal or
already-failed task. The installed-package proof image was rerun with both
cancel scenarios and printed `rx cancel race guards passed`.
The sender side now mirrors that liveness behavior for requests cancelled
before a context-side `TxSession` exists: if a late `REQUEST_DATA` arrives for
a rid already present in `_pre_cancelled_rids`, `_respond_with_kv()` sends a
failed `KV_AGENT_RESULT` instead of saving the peer request for future work.
The installed-package proof image was rerun across the receiver late-success
case, receiver INIT-before-dispatch case, and sender pre-cancelled
`REQUEST_DATA` case; it printed `nixl cancel liveness guards passed`.

The final June 13 thoroughness sweep did not identify an accepted runtime
fallback or serving oracle in the HiSparse path. Remaining references to
full-HBM, FP16, or independent references are test/baseline boundaries only:
they may be used to measure correctness from outside serving, but they must not
be wired into the coordinator, transceiver, attention dispatch, kernel ABI, or
deployment configuration as an executable alternative.

The final current-head SGLang sweep (`f7041c9d`) did not change the OP-TRT
design target. SGLang's production shape is still: decode-side HiSparse,
host-pinned full KV, a small hot device buffer, raw/request-relative top-k
capture, one-request-per-block swap-in with shared-memory hit/miss/LRU, newest
token reservation, eager decode backup, and PD direct-to-host admission. The
parts to preserve are the ownership/lifecycle algorithm and the direct-to-host
admission model. The parts not to copy are SGLang's token-slot BF16/FP8 hot
layout, naive/debug top-k loader, and staging admission as an executable OP-TRT
serving alternative. OP-TRT's first runtime-reachable candidate must remain the
target model's production dense-MLA/DSA path: packed `kvarn_k2v2` BDR blocks,
FP4 Indexer/HISA state, LayerSplit owner-local prefill, typed NIXL
direct-to-host writes, explicit sink/tail resident normal-KV reads, and
fail-closed row/status propagation.

## Final Correctness Sweep

The full implementation must remain production-architecture-first. Partial code
may exist only when it is behind fail-closed startup/mapping guards and has the
same ABI shape as the final serving path. The following are hard invariants:

- Enabled serving is not allowed until the full production chain is present:
  Indexer/HISA top-k -> native block dedupe -> request-table resolve -> hot
  plan -> compact miss schedule -> stream-ordered packed KVarN host-to-hot copy
  -> post-copy metadata commit -> hot-index build -> sparse MLA KVarN-hot
  BDR/on-read dequant. Relaxing only part of this chain is a regression.
- Dense MLA cold and hot storage is packed KVarN `kvarn_k2v2`, not FP16 and not
  an FP16 staging tier.
- The Indexer/HISA path stays device-resident FP4/HISA.
- Indexer K is not moved into KVarN or host HiSparse storage.
- HiSparse uses request-relative top-k token positions from the existing
  Indexer/HISA path and maps them into selected packed KVarN hot blocks before
  sparse MLA.
- The hot tier is block-oriented for OP-TRT, even though SGLang's generic DSA
  implementation is token-slot-oriented, because OP-TRT KVarN, BDR/on-read
  dequant, paged sparse MLA, LayerSplit ownership, and NIXL page metadata all
  key on paged blocks.
- Direct-to-host is the production path: prefill writes packed dense-MLA KVarN
  records into decode host-pinned slots through typed NIXL `HISPARSE_HOST`
  writes, and decode admission waits for commit coverage.
- SGLang's naive top-k loader/debug oracle is not a model for OP-TRT serving.
  Any offline references used by tests must stay outside the coordinator,
  transceiver, kernel ABI, and deployment config.
- There is no HELIX, completed-prefill staging fallback, full-HBM fallback, or
  direct-to-host-off fallback when `hisparse_enabled=true`.
- There is no FP16 block-hot oracle in enabled serving.
- There is no runtime downgrade when `hisparse_enabled=true`.
- LayerSplit owner-local prefill, TP4/EP4 decode, SMC-SD row expansion,
  Moondream pinning, request pinning, cancellation, and request recycle must
  compose with HiSparse before the startup guard is relaxed.
- MORI-IO remains an A/B candidate only; the gate path is NIXL write-mode/direct
  host writes.
- Promotion requires live VM proof and A/B data, not just unit tests.
- Request-table publication, host-write commit coverage, row status, and hot
  residency must be generation-checked. A recycled request id, recycled block
  id, rejected speculative branch, failed write, or stale commit generation
  must be unselectable before sparse MLA sees the row.
- The native BDR source writer must be current-stream ordered with the dense
  MLA append/commit path, and typed NIXL source reads must be ordered after the
  BDR record write. Marking a BDR record committed before the write is visible
  to the transfer source is a correctness bug.
- Fused sparse MLA KVarN-hot decode must validate the production BDR ABI before
  launch: every hot record must contain the full 2-bit C-KV, C-KV scale/zp, and
  E4M3 RoPE payload for a 64-token, 512+64 latent block.
- Sparse-MLA hot descriptors are layer-local. Reusing a descriptor across
  layers, rows, TopK widths, devices, or request steps is a stale-hot-slot
  correctness bug. The descriptor must carry the coordinator `step_id`, and the
  fused dispatch must reject any descriptor whose step id does not match the
  current coordinator step.
- Per-row planning width is the smaller of TopK width and hot capacity. Hot
  tiers larger than TopK are valid configurations and must not be rejected by
  the native block-dedupe stage; the full hot capacity still remains available
  to the LRU/hit/miss planner.
- Native mapping accepts only 2-D int32 CUDA TopK tensors whose row count
  matches cached DSA request-row geometry. A mismatched row set would bind
  TopK rows to the wrong request ids and must fail before native planning.
- The hot planner counts unique missing `(host_slot, commit_gen)` pairs per
  row. Duplicate references to the same committed block in a row must reuse the
  first planned hot slot instead of consuming another victim or false-failing
  capacity checks.
- The miss-copy boundary must be explicit. CUDA kernels must not pretend that
  CPU pinned host KVarN storage is ordinary device memory. The production path
  dedupes and plans misses on device, then hands a compact miss schedule to a
  native stream-ordered copy bridge and only then commits hot metadata.
  Python-side token extraction, Python request-table extraction, synchronous
  schedule reads, and CUDA host callbacks that enqueue CUDA work are not valid
  serving paths. The current bridge is a mapped pinned-host kernel path; a
  future copy-engine variant may replace it only if it consumes the same compact
  device schedule without host synchronization.
- Host and hot packed tiers must have non-overlapping records: last-dimension
  bytes are contiguous, slot stride covers a full packed record, and layer
  stride covers all slots in the layer. A strided aliasing view is not a valid
  HiSparse copy source or destination.
- Sparse MLA KVarN-hot consumption has the same packed-layout requirement, and
  its `stride_factor` must cover every layer's token range. Otherwise hot
  global indices can decode into the wrong hot slot or layer.
- Fused sparse MLA must fail closed at producer load too: scoring validates
  decoded hot indices before key producer loads, and the value phase performs a
  row-wide hot-index preflight before any value producer load touches
  `hot_packed`. An unexpected value-phase invalidation writes zero output plus
  `-inf` LSE instead of dereferencing a stale slot.
- Compact copy schedules must fail closed. Any impossible compact row id means
  the schedule is corrupt and no row in that batch may publish hot metadata.

Runtime/test boundary:

- Runtime-reachable HiSparse code may only use production-shaped data:
  `kvarn_k2v2` BDR records for committed dense MLA blocks, the live normal KV
  path for resident sink/tail tokens, FP4/HISA Indexer state, NIXL direct
  host-write metadata, and native row-status propagation.
- Tests may allocate synthetic tensors only to exercise the production ABI.
  A synthetic tensor must have the same byte layout, strides, status semantics,
  and launch contract as the serving path. Synthetic FP16 hot pools, naive
  token loaders, or all-hot correctness shortcuts cannot be called from DSA,
  the coordinator, the transceiver, or deployment manifests.
- Baselines may compare against the existing production full-HBM KVarN path,
  but that comparison path remains external to enabled HiSparse. It cannot be
  offered as a fallback after a HiSparse readiness, row-status, copy, or
  resident-token failure.
- A microkernel is acceptable only if it is a real producer-load implementation
  of the production layout. A slower direct per-row/head sparse MLA kernel can
  be a first runtime-reachable candidate if it consumes packed `kvarn_k2v2`
  records, row status, and explicit sink/tail descriptors.
- A pre-dequant "correctness" kernel or FP16 block-hot path cannot be a serving candidate.

June 13 final sweep result:

- The current runtime-reachable HiSparse code is still fail-closed until the
  full production chain is present. The remaining "fallback" references in the
  HiSparse path are prohibitions, error messages, or test/benchmark boundaries,
  not alternate serving implementations.
- The only acceptable first serving candidate is the target model's production
  dense-MLA/DSA path with packed `kvarn_k2v2` BDR hot records, native
  planner/copy, explicit sink/tail normal-KV resident reads, NIXL direct
  host-write admission, row-status propagation, and LayerSplit/request-pinning
  composition. A microkernel that uses this exact ABI may be slow initially,
  but it is still the production path; an FP16, NVFP4, full-HBM, or synthetic
  block-hot kernel is not.
- The branch now has native-op smoke proof for the production-shaped CUDA
  planner/copy chain and for `libth_common.so` loaded both mounted and
  image-internal with its native siblings. The remaining proof is serving
  packaging: install the full branch Python package and `libth_common.so` in
  the deployment image site-packages exactly as DSA imports them, then run the
  same production-ABI smoke through that path before any readiness guard is
  relaxed.

CUDA API note: NVIDIA documents `cudaMemcpyBatchAsync()` as a host API over
host-visible source pointer, destination pointer, and size arrays, and documents
that `cudaLaunchHostFunc()` callbacks must not make CUDA API calls. Therefore a
callback that waits for device schedule compaction and then enqueues copy-engine
work would be invalid, and a schedule readback before copy submission would
violate the no-sync production path. CUDA also documents mapped registered host
memory as device-addressable when the device supports
`cudaDevAttrCanUseHostPointerForRegisteredMem`; the current bridge uses exactly
that stream-ordered mapped-host path and fails closed otherwise.

## Relevant SGLang Facts

SGLang HiSparse is decode-side hierarchical memory for DSA/DSv4 models. The
guide states that prefill model execution is transparent, decode keeps only a
small hot device buffer, and the complete KV lives in CPU pinned memory. In PD
mode, prefill writes KV directly to decode host memory via RDMA. For DeepSeek
V4, SGLang writes only C4 KV to host and keeps the indexer/C128 path
device-to-device. In OP-TRT, "prefill transparent" means no target-model compute
detour: the prefill transceiver still has to expose/write NIXL host descriptors
for the production packed KVarN host tier.

Implementation details worth preserving or adapting:

- `HiSparseTokenToKVPoolAllocator` separates logical capacity from hot device
  capacity. `alloc_logical_only()` is used by direct-to-host transfer.
- `HiSparseCoordinator` owns request-to-host rows, request hot buffers,
  `full_to_hisparse_device_index_mapping`, LRU slots, raw top-k capture buffer,
  graph-safe output buffers, request-admission queues, eager backup stream, and
  cleanup.
- `swap_in_selected_pages()`/`load_cache_to_device_buffer_*` launches one CUDA
  kernel per layer and returns device locations for attention.
- The CUDA kernel has a short-sequence fast path, newest-token reserved slot,
  shared-memory top-k hash, LRU hit/miss compaction, host-to-device miss copy,
  and `num_real_reqs` early exit for padded CUDA-graph batches.
- DSv4 top-k captures raw request-relative token positions separately from
  physical page-table locations so the swap-in path can target logical host
  rows.
- SGLang validates model/backend constraints, requires radix cache disabled for
  HiSparse, and pairs DSA backends with the selected KV dtype.

Implementation details not to copy directly:

- SGLang's generic DSA hot buffer is token-slot based; OP-TRT's first serving
  candidate must be block-based packed KVarN.
- SGLang's BF16/FP8 FlashMLA hot tier is not OP-TRT's dense MLA KVarN hot tier.
- SGLang's staging and naive debug loader are useful for understanding
  correctness, but they are not acceptable OP-TRT deployment modes.
- DeepSeek V4 C4 layout handling is relevant as an example of architecture
  specialization, not as the OP-TRT target path for the current dense-MLA model.

## OP-TRT Facts That Change The Design

Current production r20 already enables:

- NIXL Python/native generation-first handoff;
- LayerSplit TP2xCP2 owner-local prefill;
- TP4/EP4 decode with `enable_attention_dp=true`;
- dense MLA latent KVarN `kvarn_k2v2` plus amortized restore;
- FP4 Indexer K;
- `indexcache-hisa`, HISA page reps/counts, FSSS reuse, cross-step IndexCache;
- CuTe/C++ top-k and paged MQA logits;
- WarpDecode forced on decode;
- SMC-SD with GLM draft path.

Target-model assumption for this branch: the serving path is the current
production DSA/dense-MLA target path, with `index_topk=1024`,
`tokens_per_block=64`, FP4 Indexer K/HISA, dense MLA latent KVarN
`kvarn_k2v2`, sparse MLA decode, NIXL Python/native generation-first handoff,
LayerSplit owner-local prefill, and TP4/EP4 decode. Draft-model GQA KVarN is a
separate fail-closed path and should not determine the first HiSparse design.

The clean OP-TRT insertion point is in
`DSATrtllmAttention.sparse_attn_predict()` after `Indexer.forward()` has filled
local request-relative top-k and before local top-k is converted to global pool
indices. The current transform path is:

1. `Indexer.sparse_attn_indexer()` produces `topk_indices_buffer` as
   request-relative token positions.
2. `transform_local_topk_reuse_or_compute()` either converts those positions to
   global pool indices or applies FSSS affine reuse.
3. `sparse_attn_predict()` uses global indices to trigger LayerSplit dense-KV
   read-set broadcast, then sparse MLA reads the global indices.

HiSparse should hook between steps 1 and 2 for owning F layers, and must provide
equivalent cached output for reuse S layers.

## Architecture

Add an `OPTRTHiSparseCoordinator` owned by `DSACacheManager` and exposed through
`DSAtrtllmAttentionMetadata`.

The coordinator owns:

- logical request rows:
  - `req_to_logical_blocks[req_pool_idx, block_pos] -> logical/full block id`
  - `req_to_host_blocks[req_pool_idx, block_pos] -> host block slot`
  - `req_to_hot_blocks[req_pool_idx, slot] -> hot device block slot`
- per-layer hot state:
  - `hot_block_tokens_or_block_pos[layer, req_pool_idx, slot]`
  - `hot_block_locs[layer, req_pool_idx, slot]`
  - `lru_slots[layer, req_pool_idx, hot_slot]`
- KVarN state:
  - host KVarN packed blocks;
  - host mirrors for `valid`, `commit_gen`, and restored epochs;
  - newest/tail fp16 blocks that are not committed yet;
- graph-safe buffers:
  - `hot_global_indices_buffer[B * next_n, index_topk]`;
  - `hot_block_ids_buffer[B * next_n, ceil(index_topk / tokens_per_block)]`;
  - `num_real_rows`;
  - optional `miss_count`, `hit_count`, and debug counters;
- lifecycle:
  - direct-to-host admission;
  - eager backup after decode;
  - abort/retract cleanup;
  - block recycle invalidation;
  - request-pin cleanup integration.

The hot buffer should be sized in blocks, not tokens:

```text
index_topk = 1024 tokens
tokens_per_block = 64
selected blocks/request/layer <= 16, usually fewer after block dedupe

candidate defaults:
  hisparse_hot_blocks_per_req = 64
  hisparse_hot_tokens_equivalent = 4096
  A/B: 32, 64, 96, 128 hot blocks/request
```

This block-level design matches OP-TRT's sparse MLA global-index format,
LayerSplit read-set broadcast, KVarN commit/restore, and NIXL descriptors. It
also reduces LRU pressure because many top-k tokens share a block.

## Config Surface

Add these fields under `sparse_attention_config`:

```yaml
hisparse_enabled: false  # schema default while gated; promotion candidate sets true
hisparse_mode: dense_mla_kvarn
hisparse_direct_to_host: true
hisparse_indexer_host_tier: false
hisparse_topk: 1024
hisparse_hot_blocks_per_req: 64
hisparse_host_to_device_ratio: 8
hisparse_min_seq_len: 65536
hisparse_block_lru: true
hisparse_eager_backup: true
hisparse_fail_closed: true
```

Validation:

- `hisparse_topk` must equal `index_topk` unless an explicit override is tested.
- `hisparse_mode=dense_mla_kvarn` requires `mla_latent_kv_dtype=kvarn_k2v2` or
  another supported dense MLA KVarN dtype.
- `hisparse_indexer_host_tier=false` in the first production candidate.
- decode `kv_cache_config.enable_block_reuse` must remain false unless a
  HiSparse-aware reuse adapter is implemented.
- direct-to-host requires `cache_transceiver_config.backend=NIXL` and
  `transceiver_runtime=PYTHON`.
- `hisparse_enabled=true` production manifests must set
  `hisparse_direct_to_host=true`; completed-prefill/full-HBM variants are
  separate baselines, not an enabled-HiSparse fallback.
- if HiSparse setup fails and `hisparse_fail_closed=true`, startup should fail
  rather than silently using full-HBM sparse attention.
- startup validation should reject FP16 host/hot serving tiers, direct-to-host
  disabled with HiSparse enabled, Indexer K KVarN selection, or any staging
  fallback marker in production manifests.

## Host Pool Layout

Use a dense-MLA host pool parallel to KVarN side-pool format. The serving path
must never allocate a plain FP16 host/hot tier for committed blocks.

For each local layer and physical block:

```text
host_kvarn_ckv_packed[block_id]
host_kvarn_kpe_packed[block_id]
host_kvarn_meta[block_id]
host_block_valid[block_id]
host_commit_gen[block_id]
host_owner_request_epoch[block_id]
```

Tail/sink policy:

- sink blocks stay resident or are preloaded into the hot buffer;
- the in-progress tail block stays fp16 until it becomes full;
- once full, it is committed to KVarN host storage and invalidates any stale
  hot/restored epoch.

Correctness comparisons may use offline reference buffers or the existing
full-HBM production KVarN path, but no `hisparse_host_format=fp16` serving mode
should be added. Production targets packed KVarN host storage plus BDR/on-read
dequant.

## Swap-In Kernel

Port the SGLang algorithm into OP-TRT as a CUDA/C++ op, but make it block-based:

Inputs:

```text
local_topk_tokens: int32 [rows, index_topk]
block_table: int32 [seqs, max_blocks]
req_idx_per_row: int32 [rows]
kv_lens: int32/int64 [seqs]
req_pool_indices: int32 [seqs]
req_to_host_blocks: int64 [req_slots, max_blocks]
hot_block_tokens: int32 [layers, req_slots, hot_blocks]
hot_block_locs: int32 [layers, req_slots, hot_blocks]
lru_slots: int16 [layers, req_slots, hot_blocks]
host_kvarn_pool pointers
device_hot_pool pointers
num_real_rows: int32[1]
```

Outputs:

```text
hot_global_indices: int32 [rows, index_topk]
hot_block_ids: int32 [rows, max_selected_blocks]
hit/miss stats optional
```

Algorithm:

1. One CTA per row or per `(row, layer)` depending measured occupancy.
2. Convert each selected local token to its logical block position:
   `block_pos = token // tokens_per_block`, `offset = token % tokens_per_block`.
3. Deduplicate selected block positions in shared memory.
4. Short path: if `seq_len <= hot_blocks_per_req * tokens_per_block`, map tokens
   directly to preloaded hot blocks.
5. Long path:
   - hash selected block positions;
   - scan LRU slots for hits;
   - assign misses to evictable slots;
   - update LRU order;
   - emit a compact miss schedule for packed host KVarN block to hot KVarN
     block copies;
   - submit that schedule to a native stream-ordered copy bridge. The current
     bridge uses a checked mapped pinned-host CUDA kernel path and fails closed
     when the host tier is not device-addressable; a future copy-engine variant
     may replace it only if it consumes the same compact device schedule without
     Python materialization or a synchronous host readback;
   - commit hot metadata only after copy submission succeeds;
   - update `hot_global_indices` so sparse MLA reads from hot physical block ids
     plus original token offset.
6. Latest token:
   - reserve a hot tail slot like SGLang's newest-token slot;
   - update it from the decode append path;
   - do not evict it until committed/backed up.
7. CUDA graph:
   - no allocations;
   - no Python token/request-table reads;
   - no synchronous host schedule readback;
   - no unguarded CUDA-kernel dereference of CPU pinned KVarN storage. Mapped
     pinned-host reads are allowed only through the checked native copy bridge;
   - `num_real_rows` guards padded graph rows;
   - fixed buckets for hot block count and top-k.

Initial kernels to compile:

```text
SM100, index_topk=1024, tokens_per_block=64
hot_blocks_per_req in {32, 64, 96, 128}
row shapes: B in graph buckets, next_n in {1, 2, 3, 4, 1 + gamma}
```

## KVarN Integration

Current KVarN-only serving can restore committed dense MLA blocks into the main
decode pool before attention. That existing path is a baseline and compatibility
reference, not the HiSparse implementation target. HiSparse must move committed
production serving blocks to two packed tiers:

1. cold host tier: packed KVarN records for full committed blocks;
2. hot device tier: packed BDR/KVarN records for selected committed blocks.

Live sink/tail tokens that are not yet committed full blocks remain the normal
production append/sink responsibility and must be handled explicitly by the
sparse MLA ABI. They are not a HiSparse FP16 hot tier, not an oracle, and not a
staging substitute for committed cold blocks.

The optimized target is:

- host-to-hot copies packed KVarN records;
- sparse MLA reads through a hot-pool view;
- BDR/in-kernel dequant-on-read handles selected hot blocks;
- the sparse MLA hot-read layout is the production BDR low-bit dense-MLA
  layout. It must not accidentally consume the older Python/Sinkhorn
  `KVarNLatentPool` record shape unless that record shape has first been
  intentionally migrated or adapted into the production BDR ABI;
- `commit_gen` and hot residency generations remain block-id keyed;
- `KVarNBDRSourcePool.invalidate_blocks()` is called for HiSparse BDR source
  records, and legacy `KVarNLatentPool` invalidation remains separate, when
  `free_resources()` or `rewind_kv_cache()` recycles a block id.

Implementation sequence:

1. Install production packed host/hot KVarN allocation and metadata first.
2. Implement packed KVarN host-to-hot swap-in and hot global-index mapping.
3. Define and implement the sparse MLA KVarN-hot ABI against the target
   production BDR layout, not a compatibility wrapper around the existing
   NVFP4 sparse MLA ABI:
   - `q`: bf16 `[B, s_q, 128, 576]`;
   - `hot_packed`: uint8 layer-major hot records;
   - `hot_indices`: int32 `[B, s_q, topk]` using the hot-slot global-index
     contract emitted by `hisparse_build_hot_indices`;
   - sink/tail descriptors for live resident tokens that are not yet committed
     full blocks, explicitly separated from packed committed host/hot blocks;
   - row status from resolve/plan/copy/commit/build, consumed before attention
     so invalid rows cannot read stale hot slots;
   - explicit BDR field offsets/strides, `tokens_per_block`, `kvarn_bits`,
     and dense MLA dimensions, derived from the configured production model
     rather than inferred from legacy test tensors.
4. Make sparse MLA consume the hot packed KVarN view through BDR/on-read dequant
   in the producer load path. This must be a real KVarN-hot producer read:
   decode hot global index -> `(hot_slot, token_offset)` -> BDR byte fields ->
   2-bit C-KV dequant with byte-addressed scale/zp -> inverse 128-wide BDR
   readback to the original dense MLA latent frame -> RoPE payload read ->
   existing sparse MLA math/combine where compatible. Reuse scheduler/combine
   pieces only where their memory-layout assumptions still match the
   KVarN-hot ABI. The optimized version may fold the inverse BDR algebra into
   query/value accumulation, but it must remain mathematically equivalent to
   the native packed BDR reader and must not introduce an intermediate dense
   hot staging tier.
5. Keep external FP16/KVarN references in tests only; do not add a serving
   staging path that dequants committed cold blocks into a hot FP16 pool, and
   do not add an executable "correctness" placeholder that can answer requests
   before the production KVarN-hot reader is live.

## Production Sparse MLA KVarN-Hot Contract

The next unlock is not a nominal op name. It is the production dense-MLA
hot-read path for the target model architecture. A valid implementation must
satisfy all of these conditions before the HiSparse startup/mapping guard can
be relaxed:

- The serving kernel path reads packed `kvarn_k2v2` hot records, not FP16,
  BF16/FP8 FlashMLA, NVFP4 sparse MLA records, or the legacy Python/Sinkhorn
  `KVarNLatentPool` layout.
- The op schema carries the real BDR layout fields: C-KV low-bit byte offsets,
  scale/zp offsets, RoPE byte offsets, record stride, `tokens_per_block`,
  `kvarn_bits=2`, dense MLA head dimensions, and layer/hot-slot strides.
- Row status from native block dedupe, request-table resolve, hot-slot plan,
  miss compaction, copy submission, post-copy commit, and hot-index build is
  reduced into a device-visible launch/row validity contract. Invalid rows
  cannot be masked into plausible output after reading stale hot storage.
- Sink and tail tokens that are still resident in the normal decode KV path are
  represented explicitly in the ABI. They are not modeled as a committed packed
  host block and are not copied through an FP16 hot tier.
- The producer-load path performs the BDR fold in-kernel on read. A separate
  pre-dequant pass into a dense hot buffer is a serving placeholder and is not
  acceptable for promotion.
- Correctness proof compares the production HiSparse hot path against the
  existing production full-HBM `kvarn_k2v2` path within KVarN tolerance. An
  independent FP16 reference may exist only as an offline fixture outside the
  coordinator, transceiver, kernel ABI, and deployment config.
- The implementation is wired through the current production DSA/dense-MLA
  path with Indexer/HISA, FSSS reuse, LayerSplit owner-local prefill,
  NIXL direct-to-host admission, request pinning, Moondream pinning, and
  SMC-SD row expansion. It is not a standalone microkernel promotion.
- The first acceptable CUDA target is SM100/B200 with the production buckets:
  `index_topk=1024`, `tokens_per_block=64`, and hot blocks/request
  `{32,64,96,128}`.

## Indexer And HISA Integration

HiSparse must not perturb scoring.

Keep these paths unchanged:

- FP4 Indexer K cache;
- HISA candidate page reps/counts;
- HISA min-seq gate;
- CuTe/C++ top-k dispatch policy;
- FSSS `index_topk_freq`;
- cross-step IndexCache reuse;
- short-sequence indexer skip.

Add:

- `metadata.hisparse_coordinator`;
- `metadata.hisparse_local_topk_cache`;
- `metadata.hisparse_hot_global_idx_cache`;
- `metadata.hisparse_hot_block_ids_cache`;
- per-step cache invalidation in `prepare()`, `on_update_kv_lens()`, and
  `update_for_spec_dec()`.

F layer:

1. consume local request-relative top-k from Indexer/HISA;
2. run block swap-in;
3. produce hot global indices;
4. cache hot global indices for S layers.

S layer:

1. skip scoring as today;
2. reuse cached selected local tokens or hot global indices;
3. apply layer offset only if hot block layout is layer-contiguous in the same
   way as the full pool. If each layer has independent hot block slots, reuse
   must rerun only the cheap mapping for that layer, not the top-k scoring.

The second option is safer: keep FSSS scoring reuse, but run the per-layer
swap-in/mapping because each layer's hot residency differs.

## LayerSplit Composition

Prefill:

- LayerSplit owner-local remains on.
- Owner CP ranks own full Indexer K and dense/KVarN blocks for their layers.
- Direct-to-host NIXL writes must publish global layer metadata, preserving the
  existing owner-local transfer fix.
- All CP ranks continue participating until partial-rank transfer is explicit.

Decode:

- Current r20 decode has CP1, so HiSparse decode can be local per TP rank.
- If CP decode is enabled later, swap-in should happen on the layer owner and
  the selected hot blocks should be broadcast to peer CP ranks.

Important ordering:

1. Indexer K LayerSplit broadcast still happens before scoring.
2. HiSparse swap-in happens after top-k selection.
3. LayerSplit dense-KV broadcast must use the hot selected block ids, or be
   bypassed when decode CP1 owns the hot pool.
4. Sparse MLA consumes hot global indices.

## NIXL Direct-To-Host

Use Dynamo/NIXL write-mode semantics:

- decode creates writable descriptors for host-pinned HiSparse pool regions;
- decode publishes `RdmaMetadata` through the existing generation-first
  `ctx_info_endpoint`/request pin path;
- prefill creates write operations using local descriptors and remote writable
  metadata;
- transfer begins immediately and decode admits the request when the write
  completes.

OP-TRT changes:

1. Publish HiSparse host-pinned pool metadata through `RankInfo` separately
   from the existing GPU page-table metadata.
2. Register HiSparse host-pinned pools with the transfer agent as `DRAM`
   descriptors, not as part of the existing `VRAM` KV-cache descriptor set.
3. Add a dedicated HiSparse host-write meta path, or another explicit
   `DRAM`-typed write batch, before scheduling prefill writes into host slots.
   The current `WriteMetaType.KV` path is VRAM-only and must not silently mix
   host-pinned descriptors into a GPU KV transfer.
4. Add request-level host slot allocation before `prepare_context_requests()`
   promotes a generation-first context request.
5. Include host block rows in request-pin/sideband metadata so prefill writes
   exact destination offsets.
6. Keep cancel behavior strict: if any NIXL task is mid-write, do not free host
   or hot slots until `cancel_request()` reports safe.

Fallback policy:

- Direct-to-host failure should fail closed for production.
- No staging/debug fallback should exist in the serving path. Unit tests may
  inject synthetic host-pool contents directly, but deployment config should
  expose only the production NIXL direct-to-host path.

## SMC-SD And Moondream Decode

SMC-SD changes the row geometry. HiSparse must treat speculative rows as query
rows over the same request host table.

Requirements:

- `num_real_rows` is `num_generations * next_n` or the expanded MTP row count.
- `req_idx_per_row` maps every speculative row back to the base request.
- selected top-k token positions are request-relative, not draft-row-relative.
- newest/tail backup is committed-token aware:
  - back up accepted target tokens;
  - do not commit rejected draft branches into the host table;
  - keep draft model GQA KVarN separate and fail-closed until proven.
- Moondream pinning markers must remain tied to the same `disagg_request_id`,
  `ctx_dp_rank`, and `ctx_info_endpoint`.

Do not gate first HiSparse implementation on GQA KVarN. Dense MLA target
HiSparse is the priority path.

## Scheduling And Admission

Admission should be governed by separate capacities:

```text
logical_host_capacity_blocks
hot_device_capacity_blocks
request_slot_capacity
metadata_buffer_capacity
NIXL writable-session capacity
```

For a new decode request:

1. reserve request slot;
2. allocate logical host rows for all prompt blocks;
3. allocate or reserve hot blocks for short-sequence preload/newest slot;
4. publish writable host descriptors through request pin metadata;
5. prefill writes host KVarN blocks;
6. decode admits request and preloads short sequences if needed;
7. first decode step skips eager backup for prefill tokens already in host.

Retraction:

- wait for pending backup;
- cancel NIXL session;
- if mid-write, defer free;
- clear hot maps;
- free hot slots;
- invalidate KVarN host/hot records for recycled blocks;
- free host slots;
- clear request pin metadata.

## Tests

Unit tests:

- allocator separates logical host and hot device capacity;
- host slot allocation and cleanup restore all counters;
- direct-to-host admission does not allocate full GPU KV;
- short sequence preload maps exact token offsets;
- long sequence hit/miss/LRU against a pure reference model of the hot-block
  state machine;
- newest-token reserved slot;
- duplicate top-k tokens and duplicate blocks;
- padded CUDA graph rows via `num_real_rows`;
- abort while NIXL write is transferring;
- request recycle invalidates KVarN host/hot records;
- FSSS S layers reuse scoring but still map per-layer hot slots.

Correctness tests:

- Indexer selected SET unchanged with HiSparse off/on.
- HISA candidate selection unchanged.
- sparse MLA output within KVarN quant tolerance versus the existing production
  full-HBM KVarN path.
- production full-HBM KVarN decode path vs HiSparse packed-hot block
  equivalence within KVarN tolerance. Any full-restore baseline is external to
  the enabled HiSparse serving path.
- independent offline references may be used only as fixtures that are outside
  serving. They may read recorded production-layout tensors after a run, but no
  FP16 block-hot oracle path may be wired into the coordinator, transceiver,
  attention dispatch, kernel ABI, or deployment config.
- LayerSplit TP2xCP2 prefill to TP4/CP1 decode E2E.
- generation-first NIXL direct-to-host E2E.
- streaming cancel/cleanup E2E.
- SMC-SD speculative rows with accept/reject cleanup.

Performance tests:

- swap-in microbench:
  - hit rate sweep;
  - hot blocks per request sweep;
  - top-k 1024 and 2048;
  - seq len 1k, 4k, 16k, 32k, 64k, 128k;
  - B 1, 4, 8, 16, 32, 64.
- NIXL direct-to-host:
  - host registration cost;
  - descriptor coalescing on/off;
  - LIBFABRIC vs UCX plugin;
  - CPU NUMA placement for host pool.
- end-to-end:
  - target concurrency 16;
  - input lengths 1k to 128k;
  - compare full-HBM sparse attention, production KVarN only, HiSparse packed
    KVarN, and HiSparse packed KVarN with direct-to-host.

## Production Workstreams And Gates

These labels describe ordered workstreams and proof gates. They are not
deployable states. Any incomplete workstream remains fail-closed and must not
become a serving candidate until all promotion gates pass.

### Gate 0: Branch And Docs

- Work branch: `op-trt-hisparse`.
- Keep production r20 manifests unchanged until proof gates pass.
- Add this plan and keep a running implementation checklist.

### Gate 1: Metadata And Allocator

Files:

- `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
- new `tensorrt_llm/_torch/attention_backend/sparse/hisparse.py`
- `tensorrt_llm/llmapi/llm_args.py` or relevant sparse config parser

Deliverables:

- config parse/validation;
- coordinator object creation;
- graph-safe buffers;
- request lifecycle hooks;
- no-op disabled path;
- fail-closed startup validation.

Current branch status:

- implemented config fields and sparse-config validation for
  `hisparse_enabled`, `hisparse_mode`, direct-to-host, Indexer device
  residency, KVarN dense MLA storage, hot-block sizing, eager backup, and
  fail-closed policy;
- implemented outer runtime validation requiring
  `cache_transceiver_config.backend="NIXL"`,
  `transceiver_runtime="PYTHON"`, and
  `kv_cache_config.enable_block_reuse=false`;
- added `OPTRTHiSparseCoordinator` as the DSA-owned extension point;
- wired coordinator ownership into `DSACacheManager`, per-step metadata reset,
  and the `sparse_attn_predict()` TopK mapping seam;
- disabled HiSparse remains a no-op and preserves current behavior;
- enabled HiSparse intentionally raises before serving until packed KVarN
  host/hot allocation, NIXL commit, host-to-hot swap-in, and sparse MLA hot-read
  support are all complete and live-validated.

### Gate 2: Production Packed KVarN Cold/Hot Tiers

Deliverables:

- host-pinned packed KVarN cold pool for committed dense MLA blocks;
- hot device packed KVarN pool for selected committed blocks;
- resident sink/tail policy for live normal decode KV blocks that are not yet
  committed full-block `kvarn_k2v2` HiSparse records;
- host/hot `valid`, `commit_gen`, request epoch, and recycle invalidation;
- sparse-attention metadata exposes hot block tables without changing Indexer
  scoring.

Current branch status:

- implemented the coordinator's packed-tier descriptor for
  `num_layers`, `tokens_per_block`, `packed_bytes_per_block`,
  logical host capacity, and hot device capacity;
- implemented request host-row reservation and release, with stable
  request-relative `block_pos -> host_slot` ownership;
- implemented a host-pinned plus device-mirrored request table keyed by stable
  coordinator table slots:
  `request_ids`, `request_block_host_slots`, `request_block_commit_gen`, and
  `request_admitted`;
- kept that request table in the final CUDA-planner ABI shape, and changed
  lifecycle publication to update host rows first, then publish the affected
  slot/row to the CUDA mirror with
  `trtllm::hisparse_publish_request_table_slots`. This removes per-cell device
  scalar writes and makes enabled CUDA HiSparse fail closed if the native
  publisher is missing;
- bounded direct `configure_packed_tiers()` request-table defaults to avoid
  quadratic host-block allocation, while production `configure_from_kv_cache_manager()`
  derives table capacity from `max_batch_size` and width from
  `max_blocks_per_seq`;
- implemented host block commit metadata with `valid`, `commit_gen`,
  `logical_block_id`, and request epoch tracking;
- implemented layer-local hot-slot metadata and LRU hit/miss selection keyed by
  `(req_pool_idx, block_pos, host_slot, commit_gen)`;
- split hot-slot handling into a non-mutating `plan_hot_blocks()` phase and a
  `commit_hot_selection()` phase so miss residency is published only after the
  future native packed KVarN copy succeeds;
- added `HiSparseSwapInPlan` and pointer helpers that produce parallel host
  DRAM pointers, hot HBM pointers, and byte sizes for packed KVarN miss blocks;
- added `trtllm::hisparse_swap_in_packed_kvarn`, a strict native thop that
  copies only packed `uint8` KVarN records from pinned host memory into the hot
  CUDA tier, coalescing consecutive slot runs when both tiers are compact. This
  CPU-slot-vector helper remains a debug/building-block path and is not the
  enabled-serving copy bridge;
- added coordinator `execute_swap_in_plan()` as a CPU-slot-vector smoke/helper
  around packed-record copies. It is not part of enabled serving: production
  mapping requires the compact device-schedule bridge,
  `trtllm::hisparse_submit_packed_kvarn_copy_schedule`, and fails closed before
  the helper can become a runtime substitute;
- added request-relative token-position planning that dedupes top-k tokens into
  paged block positions without changing Indexer/HISA scoring;
- added `trtllm::hisparse_topk_to_block_positions`, a native CUDA shared-memory
  hash dedupe primitive that maps request-relative TopK tokens to unique
  request-relative block positions, emits device overflow flags without a host
  sync, and marks overflowed rows with an invalid block count so downstream
  native status checks reject clipped hot sets fail-closed;
- added `trtllm::hisparse_resolve_blocks_to_host_slots`, a native CUDA request
  table resolver that consumes row request ids, block rows/counts, device
  request ids, block-to-host-slot rows, commit generations, and admission flags
  to produce host slots, commit generations, per-block status, and per-row
  status without host extraction;
- added `trtllm::hisparse_plan_hot_slots`, a native non-mutating CUDA planner
  that consumes resolved host slots/commit generations plus layer-local
  `hot_host_slot`, `hot_commit_gen`, and `hot_lru_tick` metadata, protects
  slots selected earlier in the same batch, and emits planned hot slots,
  planned LRU ticks, miss host/hot copy schedules, hit flags, miss counts, and
  row status without publishing hot residency;
- added `trtllm::hisparse_compact_miss_schedule`, a native CUDA schedule
  compactor that consumes planner miss tensors and row status, validates
  upstream rows/counts/slots, and emits contiguous device `host_slot` and
  `hot_slot` vectors, row ids, and a device copy count for the copy bridge;
- added `trtllm::hisparse_submit_packed_kvarn_copy_schedule`, a native mapped
  pinned-host copy bridge that consumes the compact device schedule without
  Python materialization or a synchronous schedule readback, copies packed
  KVarN records into the hot HBM tier in stream order, and returns per-row copy
  status so post-copy metadata commit can remain fail-closed;
- added `trtllm::hisparse_commit_hot_slots`, a native post-copy CUDA metadata
  commit op that mutates device `hot_host_slot`, `hot_commit_gen`, and
  `hot_lru_tick` only for rows whose native plan succeeded;
- added `trtllm::hisparse_build_hot_indices`, a native CUDA hot-index builder
  that remaps request-relative TopK token positions through selected
  request-relative block rows and planned hot slots into sparse-MLA-compatible
  hot global indices with explicit row status;
- synchronized device hot metadata (`hot_host_slot`, `hot_commit_gen`, and
  `hot_lru_tick`) whenever hot records are committed or cleared;
- implemented invalidation that clears hot records when host records are
  invalidated or request slots are released;
- added a production BDR KVarN layout descriptor and a source-layout gate:
  `configure_from_kv_cache_manager()` now requires
  `bdr_ckv_lowbit_fp8_pe_v1` source records before deriving HiSparse tier
  sizes, and rejects the current `legacy_sinkhorn_v1` KVarN side-pool rather
  than allocating host/hot buffers with the wrong ABI;
- added a production-shaped `KVarNBDRSourcePool` and DSA allocation path for
  HiSparse-enabled runs. It owns BDR byte-record storage, destination/source
  fragments, commit generations, and recycle invalidation, but deliberately
  does not add a Python FP16-to-BDR serving writer;
- added `torch.ops.trtllm.mla_bdr_write_kvarn_record`, a native production BDR
  writer that consumes the paged dense MLA latent block view and fills the
  `KVarNBDRSourcePool` record with low-bit C-KV, C-KV scale/zp, and the current
  8-bit RoPE payload;
- added `test_mla_bdr_write_kvarn_record_cuda_layout_smoke`, a tiny CUDA smoke
  test that skips unless the native writer op is loaded and then proves
  block-id targeting, BDR field writes, padding preservation, and current-stream
  completion for the production `kvarn_k2v2` BDR record shape;
- wired the KVarN full-block commit walk to populate the BDR source pool when
  the HiSparse BDR layout is active, including the backfill case where the
  legacy side-pool record was already valid but the BDR source record was not;
- added DSA writer-facing hooks:
  `kvarn_bdr_record_destination_fragments()` returns layer-major writable BDR
  record destinations for the native writer, and
  `mark_kvarn_bdr_records_committed()` publishes those records only after the
  native write succeeds;
- guarded `kvarn_packed_source_fragments()` with the same production-layout
  requirement so NIXL direct-to-host cannot transfer legacy KVarN records into
  the HiSparse host tier;
- guarded the dense MLA decode branch so enabled HiSparse cannot silently fall
  through to `sparse_mla_decode_nvfp4` or the restored full-pool TRTLLM MLA
  path before `sparse_mla_decode_kvarn_hot` exists;
- implemented production-shaped packed tensor allocation for host `uint8`
  KVarN records, device hot `uint8` KVarN records, host commit metadata, and
  device hot-slot metadata;
- wired `DSACacheManager` so an explicitly enabled HiSparse config derives
  packed tier sizes from dense MLA KVarN, allocates the host/hot tensors, and
  then still fails closed before serving until the swap-in/read kernels exist;
- added CPU-level unit tests for allocation, duplicate reservation, capacity
  failure, uncommitted-block rejection, commit-generation refresh, LRU eviction,
  non-mutating plan/commit, admitted-request enforcement, packed pointer-plan
  ABI, tensor allocation ABI, and cleanup.

Still pending before serving enablement:

- live E2E proof that the host-write completion handoff marks host `valid` and
  `commit_gen` only after typed HiSparse host writes succeed for the relevant
  layer/block coverage;
- exact-clean native thop compile and direct B200 CUDA smoke are complete for
  the planner/copy/read/write set through `th_hisparse_smoke`: TopK block
  dedupe, request-table resolve, resident-block classify, hot-slot plan,
  compact miss schedule, mapped pinned-host copy submission, post-copy hot
  metadata commit, hot-index build, BDR writer, BDR hot reader,
  sparse MLA KVarN-hot decode, resident padding, second-pass hit reuse,
  overflow rejection, unadmitted-row rejection, and uncommitted-block
  rejection. The same planner/copy smoke also passes inside the newest local
  deployment runtime image when that image loads the mounted exact-clean
  `libth_hisparse_smoke.so`, and inside a tiny proof image that contains the
  same branch-built library internally at `/opt/ai-blaise/hisparse`;
- cached branch-built `libth_common.so` native-op proof is complete: the same
  planner/copy smoke passes with `libth_common.so` mounted into buildtools,
  mounted into the deployment runtime image, and copied image-internal with
  `libtensorrt_llm.so`, `libpg_utils.so`, and decoder-attention shared-library
  siblings;
- full branch-built deployment image/wheel proof is still pending: the current
  proof images prove ABI/runtime compatibility, mapped-host access, and
  `libth_common.so` dependency closure, but they do not yet prove that the full
  `op-trt-hisparse` Python package and `libth_common.so` are installed in the
  serving image's site-packages exactly as DSA will import them;
- serving-layout import proof is complete for commit
  `c6143d16492f4dde16375eb28f68e43298e10d7e`: the proof image overlays branch
  Python into deployment-runtime site-packages, normal `import tensorrt_llm`
  loads `tensorrt_llm/libs/libth_common.so`, and the native smoke runs without
  `--library`. The heavier fullsource image remains required if
  branch-generated bindings or plugin libraries change;
- deployment-runtime proof that
  `trtllm::hisparse_submit_packed_kvarn_copy_schedule` sees the production
  host tier as mapped/device-addressable on B200 is complete for both the
  narrow exact branch-built thop library and cached branch-built
  `libth_common.so`, mounted and image-internal. If a later full branch-built
  serving image cannot use the mapped-host kernel path, replace only the bridge
  with a copy-engine implementation that consumes the same compact device
  schedule without Python materialization or synchronous host readback;
- B200 compile/live validation of `torch.ops.trtllm.mla_bdr_write_kvarn_record`
  is complete at the native-op level through `th_hisparse_smoke`: the writer
  fills the production BDR byte layout, supports byte-strided records, and the
  shared reader reconstructs the original dense MLA latent through inverse BDR
  readback. Promotion still requires proof that those current-stream writes are
  ordered before any NIXL source read in the live DSA/NIXL deployment image;
- optional coalescing of native request-table publication across multiple
  lifecycle events. Correctness no longer depends on Python scalar writes, but
  multi-slot batching may still reduce Python call overhead before promotion;
- sparse MLA hot-pool ABI and BDR/on-read dequant hookup are present and
  native-op smoke-proven; the remaining work is live DSA/NIXL/deployment proof
  plus optimized split scheduling/query-fold for throughput.

### Gate 3: Swap-In Kernel And Sparse MLA Hook

Deliverables:

- SM100 block-level swap-in kernel over packed KVarN records;
- block dedupe from local top-k token positions;
- hit/miss/LRU/newest-slot updates with graph-safe buffers;
- hot global-index output consumed by sparse MLA;
- BDR/on-read dequant for hot packed KVarN records;
- FSSS reuse layers reuse scoring but rerun per-layer hot-slot mapping when hot
  residency is layer-local.

Current branch status:

- `DSAtrtllmAttention.sparse_attn_predict()` already has the HiSparse mapping
  seam immediately after Indexer/HISA top-k production and before the existing
  full-pool index transform;
- the coordinator now exposes device-side request/admission tables that the
  next native hot-slot planner can use with device TopK block rows, without
  Python token or request-table extraction;
- attention metadata now carries `hisparse_request_ids`, keyed by
  `disagg_request_id` when present, so the decode-side planner uses the same
  request key that NIXL direct-to-host admission reserved;
- incremental update paths refresh the HiSparse request-id vector along with
  normal request ids to avoid stale admission keys under overlap/CUDA-graph
  reuse;
- runtime mapping requires configured packed tiers, allocated tensors,
  admission-compatible request ids, and the native
  `trtllm::hisparse_topk_to_block_positions`,
  `trtllm::hisparse_resolve_blocks_to_host_slots`,
  `trtllm::hisparse_plan_hot_slots`,
  `trtllm::hisparse_compact_miss_schedule`,
  `trtllm::hisparse_submit_packed_kvarn_copy_schedule`,
  `trtllm::hisparse_commit_hot_slots`, and
  `trtllm::hisparse_build_hot_indices` ops before it can proceed;
- when those ops and real CUDA metadata are present, the coordinator now runs
  the native chain through device TopK block dedupe, request-table resolution,
  hot-slot planning, compact miss scheduling, mapped packed KVarN copy
  submission, post-copy metadata commit, and hot global-index construction;
- the TopK-to-block stage now plans at `min(index_topk, hot_capacity)` so
  feasible oversized hot-tier configurations do not fail before LRU planning;
- the coordinator now validates TopK rank/dtype and row count against cached
  DSA request-row geometry before constructing native request ids;
- the native hot planner's capacity pre-check now matches its slot-selection
  pass by counting duplicate missing host/commit pairs once per row;
- added `trtllm::hisparse_read_kvarn_hot_bdr`, a native production BDR
  hot-read primitive for CUDA validation. It consumes packed hot KVarN records
  plus hot global indices and row status, dequants C-KV through the configured
  low-bit scale/zp layout, reads the E4M3 RoPE payload, and rejects wrong
  row/status/index/layer layout. It is intentionally not wired as a serving
  pre-dequant pass;
- added fused `trtllm::sparse_mla_decode_kvarn_hot` plus DSA absorption-
  generation dispatch wiring. Enabled HiSparse now routes through the typed
  KVarN-hot descriptor before the NVFP4/full-HBM path can run. The fused op
  also checks the production BDR record byte size in the host wrapper before
  launch, matching the standalone hot-reader guard. The attention dispatch
  refuses to reuse a cached descriptor unless it matches the current local
  layer, row count, row-status count, TopK width, and devices;
- tightened the schedule-driven packed KVarN copy bridge so a compact schedule
  with an invalid row id marks every row invalid, preventing post-copy hot
  metadata publication after a skipped miss copy;
- tightened the packed copy bridge tensor ABI so host/hot packed tiers must
  have non-overlapping slot and layer strides before copy submission;
- tightened the fused sparse MLA KVarN-hot wrapper so hot tier strides cannot
  alias packed records/layers and `stride_factor` covers every layer token range;
- tightened the fused sparse MLA CUDA kernel so the value phase runs a
  row-wide hot-index preflight and fails closed before any stale hot-slot read;
- the enabled-startup readiness ladder now checks native planner/copy ops,
  the standalone BDR hot-reader primitive, and fused
  `sparse_mla_decode_kvarn_hot` as separate fail-closed gates, then still
  requires the explicit sink/tail resident-token readiness probe before enabled
  serving;
- startup and mapping now hard-fail after all native sparse-MLA ops are present
  while `torch.ops.trtllm.hisparse_sparse_mla_resident_v1_ready()` returns
  false. The fused producer-load path now consumes the `explicit_sink_tail_v1`
  resident-token ABI in source, but the readiness surface remains false until
  live DSA/NIXL/B200 proof shows resident sink/tail hits, row status, stale
  descriptor rejection, and cleanup compose correctly. This prevents
  uncommitted-block row status from becoming zero-output serving behavior;
- the sparse MLA KVarN-hot descriptor now records the coordinator `step_id`
  and carries production-shaped sink/tail resident metadata:
  row kv-lens, row request ids, row request indices, the cached normal-KV
  pool view, the cached normal-KV block table source, original
  request-relative TopK token positions, sink token/block counts, tail block
  position, tail token count, and tail validity. The fused attention dispatch
  rejects same-shape cached descriptors from older request steps or descriptors
  missing matching resident-token metadata before they can consume stale
  hot-slot state;
- the `trtllm::sparse_mla_decode_kvarn_hot` native op schema now accepts the
  same `explicit_sink_tail_v1` resident-token tensors and validates them as a
  complete set when provided: row kv-lens, row request indices, row request
  ids, normal-KV pool view, normal-KV block table, original request-relative
  TopK token positions, tail block positions, tail token counts, tail validity,
  and sink token/block counts. Sink tokens must be block-aligned because the
  planner classifies resident ownership per selected block; partial sink blocks
  remain fail-closed until a per-token resident classifier exists. The
  production attention call passes these fields from the descriptor into the op.
  The CUDA kernel now uses the
  resident-read sentinel to select resident normal-KV reads for sink/tail hits;
  the separate readiness op remains false until live runtime proof is complete;
- the native planner ABI now has `trtllm::hisparse_classify_resident_blocks`,
  a CUDA classifier that consumes selected request-relative block positions,
  row kv-lens, tail block positions, tail validity, and sink-block count, then
  emits per-selected-block flags for committed-hot, resident sink, and
  resident tail. The descriptor carries those flags and their row status so
  the next planner/build stage can exclude resident blocks from host-to-hot
  copy while preserving row correctness;
- request-table resolve, hot-slot planning, post-copy commit, and hot-index
  build now consume `resident_block_flags`: resident sink/tail blocks bypass
  host-slot lookup, hot-slot victim selection, host-to-hot copy scheduling, and
  hot metadata publication, while committed-hot blocks still require admitted
  request-table host slots, valid commit generations, planned hot slots, and
  post-copy commit. Hot-index build emits `-1` as the resident-read sentinel
  for sink/tail token positions and preserves row validity for the future fused
  producer-load path;
- the fused `sparse_mla_decode_kvarn_hot` producer-load kernel now consumes
  that resident-read sentinel: committed blocks read packed `kvarn_k2v2` BDR
  records from the hot tier, while resident sink/tail hits resolve the
  original request-relative TopK token through the live normal decode block
  table and read bf16/fp16 latent K/V directly from the normal resident KV
  pool. The resident path checks row kv-lens, sink coverage, tail validity,
  block-table bounds, and resident KV-pool bounds before either score or value
  producer load can use the token. The resident reader also rejects missing or
  negative per-row request ids before resolving through the normal KV block
  table, keeping request-pinning metadata in the ABI rather than treating the
  block-table row index as sufficient identity. The fused kernel now
  distinguishes the resident-read sentinel from padded TopK entries by looking
  at the original request-relative TopK token: `hot_index < 0` plus
  `request_topk_index < 0` is padding and contributes no score/value, while
  `hot_index < 0` plus a nonnegative request token must pass the explicit
  sink/tail resident checks. Padded `-inf` scores are converted to zero weights
  before normalization so all-padding/no-sink rows cannot produce NaNs from
  `-inf * 0`;
- added
  `test_sparse_mla_decode_kvarn_hot_resident_padding_cuda_smoke`, a runtime
  proof hook that exercises `explicit_sink_tail_v1` with one resident tail
  token plus padding and a separate all-padding row. It skips until the native
  op is loaded, but once the rebuilt image is available it directly checks that
  resident reads return the normal-KV value, padded rows return zero output,
  and neither output nor LSE contain NaNs;
- the final sweep rechecked the plan and current code for stale FP16
  block-hot oracle, full-HBM serving fallback, and executable placeholder
  language. Remaining references are explicit prohibitions or external
  baseline/test-fixture boundaries. The new resident-padding smoke compiles
  locally and now passes as a direct B200 CUDA smoke against the exact-clean
  `th_hisparse_smoke` target. The first B200 proof build linked
  `libth_common.so` from the persistent dirty smoke tree, and a driver-attached
  registration probe confirmed the HiSparse thops are present. The exact-clean
  runtime checkout then exposed a real SM100-only build issue: context FMHA v2
  cubin archives can be filtered out entirely while `fmhaDispatcher.cpp` still
  includes `cubin/fmha_cubin.h`. The branch now carries a CMake-side empty
  FMHA v2 cubin metadata/header generator for that architecture-filtered case,
  and the empty metadata struct now matches the runtime source ABI, including
  the optional direct launcher pointer used by current FMHA code. The generated
  include directory is propagated to `kernels_src`, `common_src`, and the later
  `thop`/plugin build scopes that include FMHA runner headers. This is a
  build-proof fix only; it does not add an attention fallback or change
  HiSparse serving behavior. The follow-up exact-clean
  target, `th_hisparse_smoke`, links the real HiSparse registrations and
  production CUDA kernels without `common_src`/`th_common`; its direct smoke
  proved BDR writer layout, byte-strided scale/zp, inverse BDR hot read,
  sparse MLA KVarN-hot decode, and resident padding behavior. The final sweep
  added the same fused KVarN-hot smoke to the serving-import proof helper; the
  exact-clean library passes it, while the cached serving-layout
  `libth_common.so` must be rebuilt and reproved before this stronger
  serving-import gate can be closed. Full deployment proof still requires live
  DSA/NIXL metadata rather than the isolated thop script;
- if the native op, CUDA-side planner, or sparse MLA hot-pool read path is
  absent, mapping raises rather than falling back to the full-HBM transform.

Still pending before serving enablement:

- build the full deployment image/wheel with these exact-clean fixes and rerun
  the same native-op smoke through the image that DSA will load in serving.
  The exact-clean `th_hisparse_smoke` proof is complete, and the earlier
  serving-layout branch-Python + branch-`libth_common.so` proof is complete for
  the planner/copy chain. The stronger serving-layout proof that also runs
  fused `sparse_mla_decode_kvarn_hot` remains pending on a fresh
  `libth_common.so` rebuild and proof-image rerun. These proofs still avoid a
  full generated-bindings rebuild. A June 13 diff audit shows generated
  binding/plugin source did not change on this branch head, so this is not a
  current-head correctness delta; if those sources change, full generated
  artifact proof becomes mandatory before promotion;
- use `scripts/blaise_build_hisparse_thop.sh` for the current VM-side native
  thop proof loop. The June 13 build sweep established the required
  non-disruptive recipe: use a resident toolchain image, keep a persistent
  build/cache mount, pass NCCL include/library paths explicitly from the Python
  NCCL wheel and system `libnccl.so`, add the real CUTLASS FetchContent Python
  source dir to `PYTHONPATH`, disable DeepEP/DeepGEMM/FlashMLA for the
  thop-only proof,
  disable the OSS CUTLASS GEMM feature families that are not needed for this
  native-op proof, force dynamic NVRTC linking, and clear the missing `ccache`
  compiler launchers. That recipe configures cleanly for
  `BUILD_WHEEL_TARGETS=th_common` and is the fastest known non-invasive path
  to a native library containing the HiSparse `torch.ops.trtllm.*`
  registrations. The follow-up sweep parameterized the helper with
  `WHEEL_TARGETS=...` in addition to `TARGETS=...`, so the same persistent
  cache can build generated binding/plugin proof targets without a fresh
  fullsource loop. Use that only to build real CMake targets from the branch
  cache; do not introduce an ad hoc import shim or a runtime substitute for the
  generated artifacts. A broad attempt to build `bindings
  nvinfer_plugin_tensorrt_llm tensorrt_llm_transfer_agent_binding` confirmed
  that `bindings` pulls thousands of shared TRT-LLM CUTLASS/MoE objects
  (3k-plus Ninja steps) even with the persistent cache. Do not put that broad
  rebuild on the critical path unless generated binding/plugin sources changed
  or the full wheel/image gate is being run deliberately on a build lane. The
  persistent build dir is part of the recipe, not an optional convenience. The
  helper supports
  `TARGETS=...` for narrow proof builds during downtime, and the swap-in copy
  bridge now builds as `hisparseSwapInPackedKvarnOp.cu` because it owns the
  mapped-host CUDA copy kernel and launch syntax. The helper exports the CUTLASS
  FetchContent Python path inside the container shell as well as at Docker
  launch, avoiding the disabled-user-site `cutlass_library` import trap during
  repeated configure loops. The helper now forces `--entrypoint /bin/bash` so
  resident runtime images that already set `/bin/bash` as their entrypoint can
  run the build command instead of trying to execute `bash` as a script. The
  serving-import proof-image helper applies the same entrypoint rule for
  `--run-smoke`, so the proof container runs the smoke command directly rather
  than handing `bash -lc ...` back to an inherited `/bin/bash` entrypoint. On
  `a4-us-001`, the current serving-layout proof build uses
  `local/dynamo-trtllm-optrt-custom:optrt-34fe7aaec-fixed-20260611011615`
  with `/home/spencer/work/build-cache/hisparse-thop-001`; the older
  `hisa-buildtools-20260531` image was not resident there. The CUTLASS
  kernel-generation CMake step also now
  passes the FetchContent CUTLASS Python root directly into its Python
  subprocess, so native proof builds do not depend on deprecated `develop
  --user` behavior or user-site activation. The same build sweep found that
  `cublasFp4ScaledMM.cpp` was compiled even when
  `ENABLE_CUBLASLT_FP4_GEMM=OFF`, although its wrapper calls exist only behind
  that option; `th_common` now includes that thop source only when the CUBLASLt
  FP4 option is enabled, keeping the HiSparse proof build's unrelated FP4
  disable path honest. The helper still disables unrelated OSS CUTLASS
  low-latency, FP4, and all-reduce GEMM families, and `th_common` now only
  compiles `fusedGemmAllreduceOp.cpp` when
  `USING_OSS_CUTLASS_ALLREDUCE_GEMM=ON`, matching the option that builds the
  underlying runner implementation. The helper leaves
  `USING_OSS_CUTLASS_MOE_GEMM=ON` because `moeOp.cpp` expects the OSS MoE
  interface that provides `MoeGemmId`, dynamic FP4 fc2 scaling, and the current
  groupwise quantization signature. Generated SM80 CUTLASS instantiations stay
  enabled in this proof path because the OSS MoE dispatch objects still contain
  references to SM80 launchers; skipping them would require a separate,
  architecture-filtered dispatch implementation rather than a CMake-only
  shortcut;
- optimize `sparse_mla_decode_kvarn_hot` beyond the direct per-row/head kernel
  by importing only the compatible FlashMLA split scheduler/combine structure
  while preserving the packed KVarN-hot BDR producer load. The intended shape is
  to reuse the scheduler metadata and bf16 combine path, add a new KVarN-hot
  split producer that reads packed `kvarn_k2v2` records through
  `hisparseKvarnBdrRead.cuh`, writes per-split accumulators/LSE, and then lets
  the existing combine logic reduce them. Do not reuse the NVFP4 producer or
  its `[block, token, hkv, 288]` plus scale layout as a compatibility backend;
  Confucius has produced a bounded VM worktree candidate at
  `a4-us-001-rl9:/home/spencer/work/op-trt-hisparse-splitprod-wt` on branch
  `op-trt-hisparse-splitprod`. It registers
  `trtllm::sparse_mla_decode_kvarn_hot_split`, reuses only the FlashMLA sparse
  scheduler metadata, keeps packed `kvarn_k2v2` BDR producer loads and
  explicit sink/tail resident reads, and validates split output/LSE parity
  against the current direct KVarN-hot op for committed hot records, resident
  sink, resident tail, padding-only, invalid upstream row, stale hot slot, and
  stale layer cases. The June 13 audit found it promising but not merge-ready:
  scheduler/split sizing must be rechecked for per-row `topk_length`, the fake
  op metadata shape must stop being a placeholder before graph/export use, and
  the candidate needs IKP timing artifacts before it replaces the direct kernel.
  This candidate still needs main-branch audit, merge, and proof through the
  same serving-layout `libth_common.so` path before it can count as integrated
  production optimization;
- prove final row-status behavior under resolve/plan/copy/commit/build errors
  with runtime tests, including invalid-row rejection before any stale hot-slot
  read can influence output;
- live-prove CUDA producer-load consumption of the explicit sink/tail
  resident-token ABI under real DSA metadata and then flip
  `hisparse_sparse_mla_resident_v1_ready()`; until that proof lands, enabled
  serving remains fail-closed even though the production-shaped kernel path
  exists;
- live validation and microbenchmarking of native packed KVarN host-to-hot
  copy plus hot metadata update;
- runtime proof that the hot global-index output buffers and fused sparse MLA
  packed hot-pool read match the existing production full-HBM KVarN path within
  KVarN tolerance;
- FSSS reuse-layer remap over layer-local hot slots.

### Gate 4: Production Optimization Hardening

Deliverables:

- precompiled SM100 variants for the production buckets;
- no full-working-set FP16 restore for committed cold blocks;
- hit/miss telemetry, hot-buffer pressure counters, and request cleanup counters;
- performance proof that KVarN+HiSparse beats KVarN-only at long context and
  concurrency 16.

### Gate 5: NIXL Direct-To-Host

Files:

- `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`
- `tensorrt_llm/_torch/disaggregation/resource/utils.py`
- `tensorrt_llm/_torch/disaggregation/transceiver.py`
- native NIXL wrapper if memory-type descriptors are required
- Dynamo router/request-pin docs/tests if sideband metadata expands

Deliverables:

- host-pinned descriptors;
- decode writable metadata publish;
- prefill write operation into decode host pool;
- generation-first pin proof;
- cancel safety.

Current branch status:

- added `HiSparseHostTierMeta` for serializing host-pinned packed KVarN tier
  pointers, per-slot item sizes, names, layer count, host-slot count, and
  packed bytes per block;
- added coordinator helpers that expose DRAM registration descriptors for
  host `uint8` packed KVarN blocks plus host validity/commit metadata;
- added coordinator helpers that compute exact writable host-packed block
  destinations for `(layer_idx, req_pool_idx, block_pos)` without changing the
  existing Indexer/HISA or sparse MLA path;
- added idempotent request host-row reservation plus request-relative
  `hisparse_host_slots` publication in receiver request metadata;
- wired receive-session cleanup so HiSparse host rows are released on the same
  safe-close boundary as existing KV receive sessions;
- added layer-major destination-fragment construction for packed KVarN host
  writes, including bounds validation against the published host-slot capacity;
- added source-fragment construction from committed production
  `KVarNBDRSourcePool` byte records, with fail-closed rejection if the
  HiSparse source layout is still the legacy Python/Sinkhorn side pool or if
  requested records have not been committed by the native writer;
- added sender-side validation that aligns source packed KVarN block fragments
  with request-relative destination host slots for dense KV-cache pool pairs,
  skipping indexer, block-scale, and non-attention pools;
- extended native transfer metadata/request construction so typed HiSparse
  writes use distinct source and destination memory types
  (`VRAM -> DRAM` or `DRAM -> DRAM`) instead of overloading uniform KV/AUX
  descriptor assumptions;
- wired packed HiSparse `VRAM -> DRAM` host writes into the KV sender path so
  the receiver is not notified of KV success until the normal KV write and the
  HiSparse host write have both completed;
- added a backward-compatible `KV_AGENT_RESULT` commit payload that carries
  paired `(local_layer, request_block_pos)` coverage for each successful typed
  HiSparse host write;
- added receiver-side coverage accumulation and coordinator commit handoff:
  replayed coverage is idempotent, partial layer coverage remains unselectable,
  and a host block becomes globally valid only after all local layers for that
  block have been written;
- added a fail-closed receiver guard so a HiSparse-enabled request that
  reserved host slots cannot complete without successful host-write commit
  coverage;
- added explicit pending-write and admission state to the coordinator:
  request host writes begin when decode publishes host slots, finish on
  terminal host-write result, block request release while writes are pending,
  and mark a request admitted only after every reserved prompt block is
  committed;
- hardened disaggregated receive cleanup so cancelled/failed sessions are not
  treated as processable while KV/HiSparse writes are still `TRANSFERRING`, and
  `RxSession.close()` refuses to release HiSparse host rows until those writes
  reach a terminal state;
- hardened terminal receive result handling so late `KV_AGENT_RESULT SUCCESS`
  messages after `ERROR`/`CANCELLED` finish the pending host-write counter but
  do not record HiSparse commit coverage, mark host blocks valid, or admit the
  request;
- hardened the INIT-before-dispatch cancellation window so a task whose
  HiSparse host slots were reserved cannot be failed by cancel and then
  resurrected as `TRANSFERRING` by `Receiver.dispatch_task()`;
- hardened sender-side pre-cancel handling so a late `REQUEST_DATA` for a rid
  already cancelled before `TxSession` creation returns failed status instead
  of being saved as stale peer request metadata;
- extended `RankInfo` serialization so peers can publish/consume HiSparse host
  tier metadata through the existing rank-info handshake;
- extended `TransferWorker` so allocated HiSparse host tiers are registered
  with NIXL as a separate `DRAM` registration group;
- sender-side HiSparse fragments are intentionally not appended to the normal
  `WriteMetaType.KV` `VRAM -> VRAM` request. They are carried on `WriteMeta`,
  validated against request-relative host slots, and submitted as a separate
  typed `WriteMetaType.HISPARSE_HOST` request with `VRAM -> DRAM` descriptors;
- `test_hisparse_nixl_write_mode_host_transfer_source_contract` now guards the
  source-level write-mode contract: typed `HISPARSE_HOST` memory inference,
  generation-first request ids, pending-write begin/finish, success-before-result
  ordering, commit-coverage payloads, and admit-only-after-commit behavior.

Still pending before serving enablement:

- live E2E validation of the completion/commit handoff, including multi-rank
  and partial-slice cases;
- live decode-admission validation that proves request-visible host slots are
  committed before sparse MLA can select them;
- live cancel/abort testing that proves host slots remain pinned until
  in-flight DRAM writes finish across the full multi-rank NIXL path. The
  receiver-side late-success/admission guard and INIT-before-dispatch cancel
  guard, plus the sender-side pre-cancelled `REQUEST_DATA` guard, are
  implemented and runtime-proven in synthetic installed-package scenarios, but
  the multi-rank E2E cancel proof is still required;
- E2E proof that NIXL writes land directly in decode host slots before decode
  admits the request.

### Gate 6: LayerSplit, SMC, Moondream Hardening

Deliverables:

- owner-local LayerSplit direct-to-host proof;
- decode CP1 proof;
- CP>1 design guard or fail-closed validation;
- SMC row mapping;
- Moondream pin preservation;
- no draft rejected-token host pollution.

Current branch status:

- the base custom stack already has LayerSplit owner-local allocation,
  owner/peer metadata, and dense/indexer/HISA scratch routing outside HiSparse;
- HiSparse request ids are threaded through model-engine attention metadata and
  are keyed to `disagg_request_id` when available, so the planner can share the
  same request identity as the NIXL admission path;
- HiSparse itself remains fail-closed before SMC-SD or Moondream decode can
  consume host/hot records, so no speculative branch can accidentally use an
  incomplete HiSparse serving path today.

Still pending before serving enablement:

- prove LayerSplit owner-local prefill writes the exact production BDR records
  for the layers owned by each CP rank, while preserving global layer metadata
  for decode-side host slots;
- prove CP1 decode first; if decode CP is later enabled, either implement
  owner-side hot swap-in plus peer broadcast explicitly or reject that config;
- map every SMC-SD/MTP speculative row back to the base request host table and
  prove accepted target tokens, rejected draft tokens, and rollback cleanup
  update host/hot state correctly;
- prove Moondream pinning and request pinning stay bound to the same
  `disagg_request_id`, `ctx_dp_rank`, and `ctx_info_endpoint` through
  direct-to-host writes, cancellation, and cleanup.

### Gate 7: A/B And Promotion

Promotion candidate:

```yaml
sparse_attention_config:
  mla_latent_kv_dtype: kvarn_k2v2
  indexer_k_dtype: fp4
  indexer_mode: indexcache-hisa
  hisparse_enabled: true
  hisparse_mode: dense_mla_kvarn
  hisparse_direct_to_host: true
  hisparse_indexer_host_tier: false
  hisparse_topk: 1024
  hisparse_hot_blocks_per_req: 64
  hisparse_host_to_device_ratio: 8
  hisparse_min_seq_len: 65536
  hisparse_fail_closed: true
cache_transceiver_config:
  backend: NIXL
transceiver_runtime: PYTHON
kv_cache_config:
  enable_block_reuse: false
```

No promotion manifest may enable HELIX fallback, direct-to-host-off behavior,
legacy KVarN source records for HiSparse, NVFP4 sparse MLA under HiSparse, or a
full-working-set dense-MLA restore path for committed cold blocks.

A/B matrix:

- hot blocks/request: 32, 64, 96, 128;
- host/device ratio: 5, 8, 10;
- NIXL plugin: LIBFABRIC, UCX;
- direct-to-host on for every HiSparse candidate; completed-prefill/full-HBM
  paths may be measured as separate baselines, not runtime fallbacks;
- packed KVarN hot-pool ABI and BDR/on-read kernel variants only;
- TP4 vs alternate TP/EP settings;
- `free_gpu_memory_fraction` decode sweep;
- SMC on/off;
- Moondream overlap on/off.

Promotion requires:

- no correctness regression;
- no fallback logs;
- startup logs prove the packed KVarN HiSparse production path, NIXL
  direct-to-host, BDR/on-read dequant, and Indexer K device residency are active;
- no leaked request pins;
- no leaked host/hot blocks;
- no NIXL mid-write free;
- tokens/second/user improvement at concurrency 16 for long context;
- no short-context regression large enough to lower the aggregate target.

## Open Risks

1. NIXL host-pinned memory registration may need native wrapper changes because
   current memory descriptors do not encode memory kind.
2. Packed KVarN sparse MLA read needs a dedicated hot-pool ABI or an explicit
   KVarN mode; reusing the existing NVFP4 sparse MLA pointer layout would be a
   correctness bug.
3. FSSS reuse layers cannot blindly affine-shift hot global indices if each
   layer has independent hot slots. Safer first implementation reruns per-layer
   swap-in using reused local top-k.
4. SMC rejected draft branches must never be backed up to host as committed
   target KV.
5. HiSparse benefits appear mostly under high concurrency and long context.
   `hisparse_min_seq_len` should prevent low-concurrency/short-context overhead
   from hurting the default path.
6. The Python `KVarNLatentPool` layout still documents and materializes the
   earlier Sinkhorn-style record shape, while the production dense-MLA BDR path
   in `mlaKernels.cu` uses low-bit packed C-KV plus per-token/sub-block scale
   and zero point. The branch now allocates a separate
   `KVarNBDRSourcePool` for HiSparse source records and rejects legacy side-pool
   pointers at the host-write boundary. The native writer is now wired to that
   pool at full-block commit time; B200 native-op proof is complete for the
   writer, byte-strided BDR record layout, KVarN-hot reader, sparse MLA hot
   decode, and resident padding. The remaining required work is stream-order
   validation against NIXL source reads, live DSA/NIXL deployment proof, graph
   lifecycle proof, and optimized split scheduling/query-fold.

## Immediate Execution Plan

The next implementation work should continue from the current fail-closed
production ABI:

1. Promote the native planner/copy proof from thop-proof image to a full
   branch-built deployment image/wheel:
   - `th_hisparse_smoke` now compiles and loads the thops/kernels for
     top-k-to-block dedupe, request-table resolve, resident-block classify,
     hot-slot plan, compact miss schedule, mapped pinned-host copy bridge,
     post-copy commit, hot-index build, BDR writer, hot reader, sparse MLA
     hot decode, and resident padding;
   - `blaise_perf/hisparse/native_planner_copy_smoke.py` now passes on B200
     through the exact-clean buildtools image, through the newest local
     deployment runtime image with the exact-clean library mounted in, and
     through a tiny deployment-runtime proof image that contains the
     branch-built library internally. It covers overflow, unadmitted rows,
     uncommitted/stale commit generations, duplicate selected blocks, resident
     sink/tail bypass, hit/miss/LRU, copy-status propagation, mapped
     pinned-host copy, metadata commit, and hot-index construction;
   - cached branch-built `libth_common.so` now passes the same smoke when
     mounted into buildtools, mounted into the deployment runtime, and copied
     image-internal with its required native siblings;
   - serving package-loader proof is now complete: the committed
     `build_hisparse_serving_import_proof_image.sh --run-smoke` image imports
     branch Python from deployment-runtime site-packages, loads
     `site-packages/tensorrt_llm/libs/libth_common.so` through the normal
     TensorRT-LLM package loader, and reruns the native planner/copy smoke and
     fused sparse MLA KVarN-hot smoke without `--library`. The latest proof
     image for `4d274bd53338726648d9fa6c05e0d5b70375a3df` is
     `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-4d274bd53338-hisparse-serving-import-proof-20260613T142136Z`;
   - current branch-head generated binding/plugin proof is not required because
     the source diff does not touch nanobind, plugin source, or transfer-agent
     binding source. If any of those sources change later, use the persistent
     VM cache to build the remaining generated serving artifacts from the same
     branch head. The intended non-disruptive loop is:
     1. verify exact target names with the existing CMake/Ninja cache;
     2. run `scripts/blaise_build_hisparse_thop.sh` with `WHEEL_TARGETS=...`
        and `TARGETS=...` for the real generated binding, plugin, and
        transfer-agent targets;
     3. rerun `build_hisparse_serving_import_proof_image.sh --run-smoke`,
        letting it copy detected `bindings*.so`,
        `tensorrt_llm_transfer_agent_binding*.so`,
        `libnvinfer_plugin_tensorrt_llm.so`, and transfer wrappers into the
        deployment-runtime package layout;
     4. only then build or install the full `op-trt-hisparse` image/wheel and
        run the same smoke through the image that DSA will load in serving.
     If a generated target fails to build, capture the precise CMake/Ninja
     failure and keep HiSparse fail-closed; do not replace it with a Python
     import shim, stale base-image binding, or synthetic runtime proof. If the
     full branch-built image cannot use the mapped-host kernel path, replace
     only the copy bridge with a copy-engine variant that consumes the same
     compact device schedule without host sync.
2. Lock the production KVarN hot-record layout:
   - make `kvarn_k2v2` the dense MLA HiSparse source of truth;
   - align host/hot packed records with the production BDR layout used by
     `mlaKernels.cu`, including the now-fixed 2-bit read path;
   - keep `torch.ops.trtllm.mla_bdr_write_kvarn_record` locked to the
     smoke-proven C-KV, byte-addressed scale/zp, inverse-read, and RoPE byte
     contract; next proof is stream ordering against NIXL source reads in the
     deployment image;
   - reject any serving configuration that would feed the legacy
     Python/Sinkhorn `KVarNLatentPool` record layout directly into a BDR
     sparse MLA hot-read kernel.
3. Wire the sparse MLA KVarN-hot read path:
   - `sparse_mla_decode_kvarn_hot` now exists as the explicit KVarN mode and
     satisfies the production sparse MLA KVarN-hot ABI at native-op smoke
     level;
   - consume hot packed KVarN records directly through hot global indices,
     decoding `(hot_slot, token_offset)` into production BDR field addresses;
   - keep BDR/on-read dequant in the sparse MLA producer load path as the only
     serving read mode: 2-bit C-KV unpack, byte-addressed scale/zp apply,
     inverse BDR readback to the original dense MLA latent frame, and RoPE
     payload read must happen without an intermediate dense/FP16 hot staging
     pass;
   - include the shared `hisparseKvarnBdrRead.cuh` helper layer for hot-index
     decode and BDR field reads so the validation op and serving producer path
     share one production layout implementation;
   - base the first runtime-reachable serving implementation on the target
     model's production dense-MLA/DSA path and production BDR record layout,
     not on an intermediate correctness-only path;
   - propagate resolve/plan/copy/commit/build row status into attention before
     any row can read hot storage;
   - complete live proof of fused consumption of the explicit sink/tail
     resident-token ABI before flipping the resident-v1 readiness op:
     - descriptor policy string/version:
       `explicit_sink_tail_v1`;
     - per-row sink coverage derived from block-aligned
       `sink_tokens / tokens_per_block`; partial sink blocks fail closed rather
       than being truncated by integer division;
     - per-row tail block position/validity derived from the generation
       `kv_lens` visible to DSA metadata;
     - source references to the normal resident decode KV path for valid
       sink/tail hits, including the cached KV pool view and block table, not
       copies through an FP16 hot tier;
     - original request-relative TopK token positions so CUDA can distinguish
       committed-hot reads from sink/tail resident reads after hot-index
       remapping, and so padded TopK entries remain padding rather than being
       mistaken for resident normal-KV reads;
     - resident block flags from
       `trtllm::hisparse_classify_resident_blocks` are consumed by native
       resolve/plan/commit/build: sink/tail blocks bypass host-slot resolution
       and hot-copy requirements, while committed-hot blocks still require
       valid host-slot commit generations and hot-slot metadata;
     - row/token status that distinguishes committed-hot hits, resident
       sink/tail hits, uncommitted invalid hits, and stale-planner failures;
     - fused producer-load consumption that selects packed-hot BDR reads for
       committed blocks and resident normal-KV reads for sink/tail tokens
       before any row can emit output. This path is now implemented in the
       direct kernel and native-op smoke-proven on B200, but promotion still
       requires live DSA/NIXL runtime proof, performance profiling, and
       flipping the readiness op from false to true;
   - remove any need for full-working-set restore of committed cold blocks;
   - do not introduce an FP16 block-hot oracle, an NVFP4 sparse-MLA
     compatibility mode, or any executable serving placeholder while wiring
     this path. Reference comparisons stay outside serving code.
4. Prove and harden NIXL direct-to-host on live B200 VMs:
   - decode publishes writable host-pinned slots;
   - prefill writes exact packed KVarN records into those slots;
   - commit coverage is multi-rank and partial-slice safe;
   - decode admission remains blocked until all reserved prompt blocks commit.
5. Prove and harden cancellation/retraction/recycle:
   - no host or hot slot is freed while a DRAM write can still complete;
   - failed/partial writes never become selectable;
   - request recycle invalidates KVarN host/hot records and pin metadata.
6. Compose with the custom stack:
   - LayerSplit owner-local prefill and CP1 decode first, CP>1 decode guarded
     or implemented explicitly;
   - SMC-SD row geometry maps every speculative row to the base request host
     table;
   - Moondream pinning stays tied to the same `disagg_request_id`,
     `ctx_dp_rank`, and `ctx_info_endpoint`;
   - FSSS reuse keeps scoring reuse but reruns per-layer hot mapping.
7. Add production tests:
   - unit tests for block dedupe, hit/miss/LRU, commit coverage, admission,
     cancel, recycle, and FSSS reuse;
   - kernel tests for the KVarN-hot producer load path using production BDR
     records and row-status rejection, not a synthetic FP16 hot pool;
   - VM E2E for generation-first NIXL direct-to-host, LayerSplit prefill,
     TP4/EP4 decode, SMC-SD accept/reject, and Moondream pin preservation;
   - correctness comparison against the existing production full-HBM KVarN path
     within KVarN tolerance, with no FP16 serving oracle and no executable
     intermediate correctness placeholder.
8. Optimize before promotion:
   - precompile SM100 buckets for `index_topk=1024`, `tokens_per_block=64`, and
     hot blocks/request `{32,64,96,128}`;
   - optionally coalesce request-table lifecycle events into multi-slot native
     publishes if profiling shows Python call overhead is measurable;
   - tune host/device ratio, NIXL plugin, NUMA placement, graph buckets, and
     memory fraction;
   - add hit/miss, swap latency, host-write, admission wait, and cleanup
     counters.
9. Promote only after A/B:
   - target concurrency 16;
   - input lengths 1k through 128k;
   - compare full-HBM sparse, production KVarN-only, HiSparse packed KVarN with
     NIXL direct-to-host, and MORI-IO only as an A/B candidate;
   - require no correctness regression, no fallback logs, no leaked pins/slots,
     and tokens/second/user improvement on the long-context target.

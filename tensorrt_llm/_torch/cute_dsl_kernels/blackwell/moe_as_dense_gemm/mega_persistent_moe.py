# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-3 persistent decode-MoE megakernel: graph-replay-safe device
producer, dynamic work-stealing schedule, and gate+quant glue absorption on
top of the phase-2 tcgen05/TMA mainloops (mega_tcgen05.py).

Phase-3 deltas (each compile-time gated for A/B):

* ``on_the_fly`` -- the device producer, collapsed to its zero-copy form:
  the scheduler warp derives item j of the FC1-first queue directly from
  moe_sort's DEVICE outputs (tile_idx_to_expert_idx, tile_idx_to_mn_limit,
  num_non_exiting_tiles) by pure index math (j < nnet*FC1_B -> stage 0,
  t = j//FC1_B, b = j%FC1_B; else stage 1 over FC2_B). No host-built item
  table, no nv=0 padding walk: the live bound nnet*(FC1_B+FC2_B) is read on
  device per invocation (the production num_non_exiting_tiles pattern), so
  a CUDA graph captured once replays correctly for ANY routing. Remaining
  mutable state is the fc1_done counters + pop cursor + exit counter: one
  [n_tiles+2] i32 buffer that the kernel SELF-RESETS (the last CTA out,
  detected by an acq_rel exit fetch-add in the spare slot, re-zeroes it),
  so an invocation needs no host fill node at all (zeroed once at alloc;
  replaces the phase-2 two-launch eager reset and the earlier phase-3
  fill: ~9us as an eager fill, ~1.2us as a graph node -- both measured).

* pipelined pop -- the scheduler issues the NEXT item's cursor fetch-add
  at the top of the loop body and consumes it (shfl) at the tail, so the
  atomic's L2 round trip overlaps the current item's decode + dependency
  spin + dispatch instead of serializing between items. Measured B200,
  20-tile decode shape, graphed: variant C 91.7 -> 85.95us with the fill
  still in-graph; with the self-reset too, 84.77us vs the production
  2-kernel chain's 90.74us (-6.6%) -- the persistent grid now BEATS the
  chain it replaces, with bit-equal c_q/sf and outputs at the same
  requant floor (cos 0.996 vs true f32, equal to prod's own).

* ``dyn_pop`` -- CTAs pop the queue through a global atomic cursor instead
  of the static item = bidx + k*grid partition. FC1-first pop order keeps
  the phase-1 deadlock-freedom argument (an FC2 spin only waits on FC1
  items already handed to running CTAs) and is longest-item-first, so the
  greedy pop is LPT-balanced: the static partition's worst CTA carries ~196
  k-tiles vs the ~182 ideal at the 20-tile decode shape.

* ``absorb_quant`` -- FC1's A path loads the PRE-QUANT bf16 activations
  (the lowrank-gate output y) and runs the production NVFP4 quant recipe
  inline in the LDGSTS warps, materializing packed e2m1 + e4m3 sf directly
  in sA/sSFA. The glue kernel's quant epilogue and the x_q/x_sf GMEM round
  trip disappear from the layer. The gate kernel itself still launches
  BEFORE moe_sort (the router consumes its bf16 output), so only the quant
  half is absorbable in phase 3. Quant math is the bit-exact fp4_quantize
  recipe lifted from cute_lowrank_gate.py. Rows beyond mn_limit are skipped
  exactly like the production gather predicates (garbage padding rows,
  discarded by finalize).

Phase-2 base below: ONE resident grid executes the full routed chain
[FC1 (gather + blockscaled grouped GEMM) -> SwiGLU -> NVFP4 requant ->
 FC2 (blockscaled grouped GEMM) -> finalize scatter-add]
as a stream of (stage, m_tile, n_blk) items. The stage bodies are the
PRODUCTION kernels' warp-specialized pipelines, lifted verbatim from
blockscaled_contiguous_gather_grouped_gemm_act_fusion.py (FC1) and
blockscaled_contiguous_grouped_gemm_finalize_fusion.py (FC2 mainloop +
finalize epilogue), with the StaticPersistentTileScheduler replaced by the
queue walk and the FC1->FC2 dependency carried by a per-tile GMEM counter
(red.release.gpu after the FC1 tile's TMA stores complete; ld.acquire.gpu
spin in the scheduler before dispatching an FC2 item).

Structural unification that makes one SMEM/pipeline set serve both stages
(layout legality by construction -- everything below is the production
kernels' own configuration at mma_tiler (128, 256), cluster (1, 1), NVFP4):

* same tiled_mma (nvf4 128x256, CtaGroup.ONE), same k-tile = 256 elements;
* same per-stage SMEM A/B/SFA/SFB layouts (A fp4 128x256 = 16 KB, B fp4
  256x256 = 32 KB, SFA 2 KB, SFB 4 KB) -> ONE staged buffer set + ONE
  a-pipeline (LDGSTS) + ONE b-pipeline (TMA, equal tx_count both stages);
* same TMEM carve (overlapping accumulator 2 x (128x256) - 48 SF cols
  = full 512 cols) -> ONE acc pipeline, parity continues across items;
* FC2's A (the intermediate c_q) is loaded through FC1's gather-LDGSTS path
  with identity row mapping (row = m_tile*128 + r, no token map, no /topk)
  and the LINEAR [perm_m, I/16] sf contract; FC1's epilogue writes SFC in
  that same linear layout via a linear-as-atom layout view, so the
  intermediate never needs the swizzled-atom SF layout inside the grid.

Queue schema (i32 x 6 GMEM table): (stage, m_tile, n_blk, expert, nv, rsv)
  stage 0 = FC1 item: n_blk in [0, 2I/256) = 16, output cols [128*n_blk, +128)
  stage 1 = FC2 item: n_blk in [0, H/256)  = 28, output cols [256*n_blk, +256)
Static partition: CTA c executes items c, c+G, ... (G = grid size), queue
ordered FC1-first => deadlock-free (phase-1 design sec. 3 argument carries
over at pipeline granularity: a CTA's own FC1 items are fully dispatched into
its pipelines before its scheduler first blocks on an FC2 spin).
"""

from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import llvm, math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.custom_pipeline import (
    PipelineCpAsyncUmma,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    silu_f32,
    vectorized_atomic_add_bf16x8,
)


# --------------------------------------------------------------------------
# GMEM handoff primitives (validated in phase 1)
# --------------------------------------------------------------------------
@dsl_user_op
def elem_ptr(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def red_release_add_u32(ptr: cute.Pointer, val: cutlass.Int32, *, loc=None,
                        ip=None) -> None:
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(),
         cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_acquire_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value()],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def cp_async_bulk_wait_group_0(*, loc=None, ip=None) -> None:
    """Wait for ALL outstanding bulk async groups (TMA stores) to COMPLETE
    (not just .read): the writes are then globally visible, so a subsequent
    red.release.gpu publishes them (CUTLASS stream-K fixup idiom)."""
    llvm.inline_asm(
        None, [],
        "cp.async.bulk.wait_group 0;",
        "", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atom_add_ret_u32(ptr: cute.Pointer, val: cutlass.Int32, *, loc=None,
                     ip=None) -> cutlass.Int32:
    """Relaxed gpu-scope fetch-add (the dyn_pop queue cursor)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value(),
             cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def atom_acq_rel_add_ret_u32(ptr: cute.Pointer, val: cutlass.Int32, *,
                             loc=None, ip=None) -> cutlass.Int32:
    """Acq-rel gpu-scope fetch-add (the exit counter): release-orders this
    CTA's prior done[] publishes; acquire gives the last-out winner
    visibility of every other CTA's publishes before its self-reset."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value(),
             cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def shfl_idx_b32(val: cutlass.Int32, src_lane: cutlass.Int32, *, loc=None,
                 ip=None) -> cutlass.Int32:
    """Warp-wide broadcast of ``val`` from ``src_lane`` (full mask)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(val).ir_value(loc=loc, ip=ip),
             cutlass.Int32(src_lane).ir_value(loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


# --------------------------------------------------------------------------
# absorb_quant helpers: the bit-exact fp4_quantize recipe, lifted from
# cute_lowrank_gate.py (validated bit-exact vs trtllm fp4_quantize on B200)
# --------------------------------------------------------------------------
@dsl_user_op
def f32_to_e4m3_byte(val: cutlass.Float32, *, loc=None,
                     ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(val).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 lo;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $1, $1;\n\t"
            "cvt.u32.u16 $0, lo;\n\tand.b32 $0, $0, 0xff;\n\t}",
            "=r,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def e4m3_byte_to_f32(b: cutlass.Int32, *, loc=None,
                     ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 t, lo, hi;\n\t.reg .b32 h;\n\t"
            "cvt.u16.u32 t, $1;\n\tcvt.rn.f16x2.e4m3x2 h, t;\n\t"
            "mov.b32 {lo, hi}, h;\n\tcvt.f32.f16 $0, lo;\n\t}",
            "=f,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fmax_f32(a: cutlass.Float32, b: cutlass.Float32, *, loc=None,
             ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fmin_f32(a: cutlass.Float32, b: cutlass.Float32, *, loc=None,
             ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def e2m1x8_pack(f0: cutlass.Float32, f1: cutlass.Float32,
                f2: cutlass.Float32, f3: cutlass.Float32,
                f4: cutlass.Float32, f5: cutlass.Float32,
                f6: cutlass.Float32, f7: cutlass.Float32, *,
                loc=None, ip=None) -> cutlass.Int32:
    """8x f32 -> packed e2m1 i32; low nibble holds the even element
    (quantization.cuh fp32_vec_to_e2m1, the LINEAR fp4_quantize layout)."""
    vals = [
        cutlass.Float32(v).ir_value(loc=loc, ip=ip)
        for v in (f0, f1, f2, f3, f4, f5, f6, f7)
    ]
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            vals,
            "{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n\t"
            "mov.b32 $0, {b0, b1, b2, b3};\n\t}",
            "=r,f,f,f,f,f,f,f,f", has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


class MegaPersistentMoEKernel:
    """Single persistent grid: production FC1 + FC2 tcgen05 pipelines on the
    phase-1 work queue. Fixed config: NVFP4 (sf_vec 16), mma (128, 256),
    cluster (1, 1), gated SwiGLU FC1 with fp4+SFC quant epilogue, bf16
    finalize FC2."""

    def __init__(self, h: int, inter: int, num_experts: int, topk: int,
                 grid: int, skip_release: bool = False,
                 use_regalloc: bool = False, mma_n: int = 128,
                 on_the_fly: bool = False, dyn_pop: bool = False,
                 absorb_quant: bool = False):
        self.H = h
        self.I = inter
        self.E = num_experts
        self.topk = topk
        self.grid = grid
        assert mma_n in (128, 256)
        self.mma_n = mma_n
        # attribution probes (phase-2 tuning experiments)
        self.skip_release = skip_release      # drop FC1 wait_group+release
        self.use_regalloc = use_regalloc      # setmaxnreg warp repartition
        self.regs_epilog = 216
        self.regs_other = 80
        # phase-3 features
        self.on_the_fly = on_the_fly          # device producer (no item table)
        self.dyn_pop = dyn_pop                # atomic-cursor work stealing
        self.absorb_quant = absorb_quant      # inline NVFP4 quant of bf16 A
        self.dbg_quant = False                # dump inline-quant output
        # absorb staging split (debug): 1 = quant sA + cp.async sSFA,
        # 2 = cp.async sA + quant sSFA
        self.absorb_dbg = 0

        self.sf_vec_size = 16
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = False
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler = (128, mma_n, 1)  # K filled in _setup_attributes
        self.cta_group = tcgen05.CtaGroup.ONE
        self.vectorized_f32 = True
        self.occupancy = 1

        # FC1 warp map (production gather kernel), warp 11 idle (1cta).
        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.sync_transform_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = 32 * 12
        # tile_info consumers: epilog(4) + ldgsts(4) + mma + tma = 10 warps
        self.threads_wo_sched = 32 * 10

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta)
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=32 * (1 + len(self.epilog_warp_id)))
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.threads_per_warp)

        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512

        # Queue geometry.
        self.fc1_n_blocks = (2 * inter) // mma_n
        self.fc2_n_blocks = h // mma_n

    # ------------------------------------------------------------------
    def _setup_attributes(self):
        """Production FC1 _setup_attributes at the fixed config, plus the FC2
        epilogue tile. All layouts come from the production helpers."""
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0],
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype,
            self.sf_vec_size, self.cta_group, self.mma_inst_shape_mn)
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype,
            self.sf_vec_size, tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb)

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0], self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k)
        self.mma_tiler_sfa = (
            self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k // 16)
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k)
        # FC1 C space: gated -> N/2 output cols per tile.
        self.mma_tiler_c = (
            self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1] // 2,
            mma_inst_shape_k * mma_inst_tile_k)

        self.cta_tile_shape_mnk = self.mma_tiler
        self.cta_tile_shape_mnk_sfb = self.mma_tiler_sfb
        self.cta_tile_shape_mnk_c = self.mma_tiler_c

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,))
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # FC1 epilogue tile (production constant) and FC2 epilogue tile
        # (production compute at bf16 row-major); both are (128, 64) at this
        # config -- asserted host-side in the driver.
        self.epi_tile = (128, 64)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1])
        self.fc2_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, False, utils.LayoutEnum.ROW_MAJOR,
            self.out_dtype)

        # Stage counts: production formula with the (equal) per-stage AB
        # footprint; C ring uses FC1's fp4 epi staging.
        a_one = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1)
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1)
        c_one = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1)
        ab_bytes = (cute.size_in_bytes(self.a_dtype, a_one)
                    + cute.size_in_bytes(self.b_dtype, b_one)
                    + cute.size_in_bytes(self.sf_dtype, sfa_one)
                    + cute.size_in_bytes(self.sf_dtype, sfb_one))
        mbar_helpers_bytes = 1024
        # production formula: overlapped single-stage accumulator at N=256,
        # plain double-buffered accumulator otherwise
        self.num_acc_stage = 1 if self.mma_n == 256 else 2
        self.num_c_stage = 2
        self.num_tile_stage = 2
        c_bytes = cute.size_in_bytes(self.c_dtype, c_one) * self.num_c_stage
        self.num_ab_stage = (
            self.num_smem_capacity - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes
        self.num_c_stage += (
            self.num_smem_capacity - ab_bytes * self.num_ab_stage
            - (mbar_helpers_bytes + c_bytes)
        ) // cute.size_in_bytes(self.c_dtype, c_one)

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)

        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols)

        # FC1 early-release index (production formula).
        self.epi_tile_n_required = 2 * cute.size(self.epi_tile[1])
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n_required)
        # FC2 early-release index (production formula, finalize kernel).
        self.fc2_epi_tile_n = cute.size(self.fc2_epi_tile[1])
        self.fc2_iter_acc_early_release = (
            self.num_sf_tmem_cols // self.fc2_epi_tile_n)

        # FC2 finalize per-thread layout (production: bf16 -> 8-wide vectors).
        num_epilogue_threads = 32 * len(self.epilog_warp_id)
        self.fc2_ttr_racc_size = (
            cute.size(self.fc2_epi_tile[0]) * self.fc2_epi_tile_n
            // num_epilogue_threads)
        self.fc2_epi_layout = cute.make_layout(
            shape=(self.fc2_ttr_racc_size // 8, 4, 2), stride=(8, 2, 1))
        self.fc2_epi_loop_size = self.fc2_ttr_racc_size // 8
        self.fc2_element_offset = 8

        self.fc1_k_tile_cnt = self.H // self.mma_tiler[2]
        self.fc2_k_tile_cnt = self.I // self.mma_tiler[2]

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,            # x_q fp4 [orig_m, H, 1] K-major
        sfa: cute.Tensor,          # x_sf e4m3 LINEAR [orig_m, H/16, 1]
        b13: cute.Tensor,          # w13 fp4 [2I, H, E]
        sfb13_raw: cute.Tensor,    # w13 sf e4m3 flat (atom layout)
        b2: cute.Tensor,           # w2 fp4 [H, I, E]
        sfb2_raw: cute.Tensor,     # w2 sf e4m3 flat (atom layout)
        c: cute.Tensor,            # c_q fp4 [perm_m, I, 1] K-major (interm)
        sfc_raw: cute.Tensor,      # c_sf e4m3 flat -> LINEAR [perm_m, I/16]
        norm_const_tensor: cute.Tensor,   # gs_c [1] f32
        out: cute.Tensor,          # bf16 [ntok, H, 1] (finalize target)
        token_id_mapping_tensor: cute.Tensor,  # p2e [perm_m] i32
        token_final_scales: cute.Tensor,       # tfs [ntok, topk] f32
        alpha1: cute.Tensor,       # [E] f32
        alpha2: cute.Tensor,       # [E] f32
        items: cute.Tensor,        # [n_items, 6] i32 (static-list mode)
        fc1_done: cute.Tensor,     # [n_tiles + 2] i32 (zeroed; +cursor slot)
        n_items: cutlass.Int32,
        t2e: cute.Tensor,          # [n_tiles] i32 (on_the_fly)
        mn_lim: cute.Tensor,       # [n_tiles] i32 cumulative (on_the_fly)
        nnet: cute.Tensor,         # [1] i32 (on_the_fly)
        y_bf16: cute.Tensor,       # [orig_m, H, 1] bf16 (absorb_quant)
        norm_const_x: cute.Tensor, # gs_x [1] f32 (absorb_quant)
        xq_dbg: cute.Tensor,       # [orig_m, H/8] i32 (absorb debug dump)
        sf_dbg: cute.Tensor,       # [orig_m, H/16] i32->u8 (absorb debug)
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b13.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.out_dtype: Type[cutlass.Numeric] = out.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b13).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        # SFB tensors in the production swizzled-atom layout.
        sfb13 = cute.make_tensor(
            sfb13_raw.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(b13.shape,
                                                    self.sf_vec_size))
        sfb2 = cute.make_tensor(
            sfb2_raw.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(b2.shape,
                                                    self.sf_vec_size))

        # SFC in LINEAR [perm_m, I/16] expressed in the atom-shape profile
        # ((32, 4, RM), (16, 4, RK), L) so the production epilogue partition
        # code is unchanged while the bytes land row-major linear.
        k16 = self.I // self.sf_vec_size
        sfc = cute.make_tensor(
            sfc_raw.iterator,
            cute.make_layout(
                ((32, 4, c.shape[0] // 128), (16, 4, k16 // 4), 1),
                stride=((k16, 32 * k16, 128 * k16), (0, 1, 4), 0)))

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype,
            self.sf_vec_size, self.cta_group, self.mma_inst_shape_mn)
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype,
            self.sf_vec_size, tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA atoms for both stage-kinds' B/SFB (cluster (1,1): plain G2S).
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged,
                                    (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged,
                                      (None, None, None, 0))

        tma_atom_b13, tma_tensor_b13 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b13, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape)
        tma_atom_sfb13, tma_tensor_sfb13 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb13, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b2, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape)
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb2, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

        # TMA store for the FC1 intermediate.
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged,
                                      (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout,
            self.epi_tile)

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # (m_tile, n_blk, expert, valid, mn_limit, stage)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma, tiled_mma_sfb,
            a, sfa,
            tma_atom_b13, tma_tensor_b13, tma_atom_sfb13, tma_tensor_sfb13,
            tma_atom_b2, tma_tensor_b2, tma_atom_sfb2, tma_tensor_sfb2,
            tma_atom_c, tma_tensor_c, c, sfc,
            norm_const_tensor, out,
            token_id_mapping_tensor, token_final_scales,
            alpha1, alpha2, items, fc1_done, n_items,
            t2e, mn_lim, nnet, y_bf16, norm_const_x, xq_dbg, sf_dbg,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            self.c_smem_layout_staged, self.epi_tile, self.fc2_epi_tile,
            self.fc2_epi_layout,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    # ------------------------------------------------------------------
    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mA_mkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_b13: cute.CopyAtom,
        mB13_nkl: cute.Tensor,
        tma_atom_sfb13: cute.CopyAtom,
        mSFB13_nkl: cute.Tensor,
        tma_atom_b2: cute.CopyAtom,
        mB2_nkl: cute.Tensor,
        tma_atom_sfb2: cute.CopyAtom,
        mSFB2_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mCq_gmem: cute.Tensor,
        mSFC_mnl: cute.Tensor,
        norm_const_tensor: cute.Tensor,
        mOut: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        token_final_scales: cute.Tensor,
        alpha1: cute.Tensor,
        alpha2: cute.Tensor,
        mItems: cute.Tensor,
        mDone: cute.Tensor,
        n_items: cutlass.Int32,
        mT2E: cute.Tensor,
        mMnLim: cute.Tensor,
        mNnet: cute.Tensor,
        mY: cute.Tensor,
        norm_const_x_tensor: cute.Tensor,
        mXqDbg: cute.Tensor,
        mSfDbg: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        fc2_epi_tile: cute.Tile,
        fc2_epi_layout: cute.Layout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if cutlass.const_expr(self.use_regalloc):
            # warpgroup-uniform register repartition: fat epilogue (FC2
            # finalize carries 64 f32 acc/thread), lean producers/sched.
            if warp_idx <= self.epilog_warp_id[-1]:
                cute.arch.warpgroup_reg_alloc(self.regs_epilog)
            else:
                cute.arch.warpgroup_reg_dealloc(self.regs_other)

        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b13)
            cpasync.prefetch_descriptor(tma_atom_sfb13)
            cpasync.prefetch_descriptor(tma_atom_b2)
            cpasync.prefetch_descriptor(tma_atom_sfb2)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = False
        bidx, bidy, bidz = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        mma_tile_coord_v = 0
        is_leader_cta = True
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = (
            cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster))
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # --- pipelines (one set, shared by both stage kinds) ---
        a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp * 4),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True)

        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk)

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)),
            cta_layout_vmnk=cluster_layout_vmnk)

        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_wo_sched))

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr)

        # --- smem views ---
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        info_layout = cute.make_layout((6, self.num_tile_stage), stride=(1, 6))
        sInfo = storage.sInfo.get_tensor(info_layout)

        b_full_mcast_mask = None
        sfb_full_mcast_mask = None

        # --- global tiling (both stage kinds) ---
        # FC1 A = x_q [orig_m, H] (or y bf16 when absorb_quant);
        # FC2 A = c_q [perm_m, I].
        gA1_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None))
        gY1_mkl = cute.local_tile(
            mY, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None))
        gA2_mkl = cute.local_tile(
            mCq_gmem, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None))
        gB13_nkl = cute.local_tile(
            mB13_nkl, cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None))
        gB2_nkl = cute.local_tile(
            mB2_nkl, cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None))
        gSFA1_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
            (None, None, None))
        gSFA2_mkl = cute.local_tile(
            mSFC_linear_view(mSFC_mnl, mCq_gmem, self.sf_vec_size),
            cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
            (None, None, None))
        gSFB13_nkl = cute.local_tile(
            mSFB13_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None))
        gSFB2_nkl = cute.local_tile(
            mSFB2_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None))
        gToken_ml = cute.local_tile(
            token_id_mapping_tensor,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)), (None,))
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)),
            (None, None, None))
        gOut_mnl = cute.local_tile(
            mOut, cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None))

        fc1_k_tile_cnt = cutlass.Int32(self.fc1_k_tile_cnt)
        fc2_k_tile_cnt = cutlass.Int32(self.fc2_k_tile_cnt)

        # --- mma partitions ---
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgB13 = thr_mma.partition_B(gB13_nkl)
        tCgB2 = thr_mma.partition_B(gB2_nkl)
        tCgSFB13 = thr_mma_sfb.partition_B(gSFB13_nkl)
        tCgSFB2 = thr_mma_sfb.partition_B(gSFB2_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgOut = thr_mma.partition_C(gOut_mnl)

        # --- TMA partitions for B/SFB (both weight sets share sB/sSFB) ---
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        sB_grouped = cute.group_modes(sB, 0, 3)
        sSFB_grouped = cute.group_modes(sSFB, 0, 3)

        tBsB, tBgB13 = cpasync.tma_partition(
            tma_atom_b13, block_in_cluster_coord_vmnk[1], b_cta_layout,
            sB_grouped, cute.group_modes(tCgB13, 0, 3))
        tBsSFB, tBgSFB13 = cpasync.tma_partition(
            tma_atom_sfb13, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            sSFB_grouped, cute.group_modes(tCgSFB13, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB13 = cute.filter_zeros(tBgSFB13)
        _, tBgB2 = cpasync.tma_partition(
            tma_atom_b2, block_in_cluster_coord_vmnk[1], b_cta_layout,
            sB_grouped, cute.group_modes(tCgB2, 0, 3))
        _, tBgSFB2 = cpasync.tma_partition(
            tma_atom_sfb2, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            sSFB_grouped, cute.group_modes(tCgSFB2, 0, 3))
        tBgSFB2 = cute.filter_zeros(tBgSFB2)

        # --- mma fragments / TMEM layout (identical both stages) ---
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0], tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols)
                        * tCtAcc_fake.stride[0][1],
                    )))
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage))

        self.cta_sync_barrier.arrive_and_wait()

        griddepcontrol_wait()

        # ==================================================================
        # Scheduler warp: queue walk + FC2 dependency spin.
        # on_the_fly: items derived from moe_sort device outputs (the device
        # producer); dyn_pop: atomic-cursor work stealing at mDone[n_tiles].
        # ==================================================================
        if warp_idx == self.sched_warp_id:
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage)

            lane_idx = tidx % 32
            n_live = cutlass.Int32(n_items)
            n_fc1 = cutlass.Int32(0)
            if cutlass.const_expr(self.on_the_fly):
                nnet_v = mNnet[0]
                n_fc1 = nnet_v * self.fc1_n_blocks
                n_live = nnet_v * (self.fc1_n_blocks + self.fc2_n_blocks)

            item = cutlass.Int32(bidx)
            if cutlass.const_expr(self.dyn_pop):
                popped = cutlass.Int32(0)
                if lane_idx == 0:
                    popped = atom_add_ret_u32(
                        elem_ptr(mDone, (self.n_tiles_bound,)),
                        cutlass.Int32(1))
                item = shfl_idx_b32(popped, cutlass.Int32(0))

            while item < n_live:
                # Software-pipelined pop: issue the NEXT item's fetch-add at
                # the top of the body so its L2 round trip overlaps the
                # decode + dependency spin + dispatch below, instead of
                # serializing between items. The shfl at the loop tail is
                # the consume point (the register dependency stalls there,
                # by which time the atomic has long landed).
                popped_next = cutlass.Int32(0)
                if cutlass.const_expr(self.dyn_pop):
                    if lane_idx == 0:
                        popped_next = atom_add_ret_u32(
                            elem_ptr(mDone, (self.n_tiles_bound,)),
                            cutlass.Int32(1))
                stage = cutlass.Int32(0)
                m_tile = cutlass.Int32(0)
                n_blk = cutlass.Int32(0)
                expert = cutlass.Int32(0)
                mn_limit = cutlass.Int32(0)
                if cutlass.const_expr(self.on_the_fly):
                    if item < n_fc1:
                        m_tile = item // self.fc1_n_blocks
                        n_blk = item - m_tile * self.fc1_n_blocks
                    else:
                        j2 = item - n_fc1
                        stage = cutlass.Int32(1)
                        m_tile = j2 // self.fc2_n_blocks
                        n_blk = j2 - m_tile * self.fc2_n_blocks
                    expert = mT2E[m_tile]
                    mn_limit = mMnLim[m_tile]
                else:
                    stage = mItems[(item, 0)]
                    m_tile = mItems[(item, 1)]
                    n_blk = mItems[(item, 2)]
                    expert = mItems[(item, 3)]
                    mn_limit = mItems[(item, 4)]

                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                if stage == 1:
                    # FC1 -> FC2 handoff: all FC1 blocks of this m-tile
                    # must have published their stores.
                    cnt = ld_acquire_u32(elem_ptr(mDone, (m_tile,)))
                    while cnt < self.fc1_n_blocks:
                        cnt = ld_acquire_u32(elem_ptr(mDone, (m_tile,)))
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = m_tile
                    sInfo[(1, tile_info_producer_state.index)] = n_blk
                    sInfo[(2, tile_info_producer_state.index)] = expert
                    sInfo[(3, tile_info_producer_state.index)] = (
                        cutlass.Int32(1))
                    sInfo[(4, tile_info_producer_state.index)] = mn_limit
                    sInfo[(5, tile_info_producer_state.index)] = stage
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                if cutlass.const_expr(self.dyn_pop):
                    item = shfl_idx_b32(popped_next, cutlass.Int32(0))
                else:
                    item = item + gdim

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = 0
                sInfo[(1, tile_info_producer_state.index)] = 0
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = 0
                sInfo[(5, tile_info_producer_state.index)] = 0
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # ==================================================================
        # LDGSTS A/SFA warps (4-7): FC1 gather path; FC2 identity path
        # ==================================================================
        if warp_idx <= self.ldgsts_a_warp_id[-1] and warp_idx >= self.ldgsts_a_warp_id[0]:
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mA_mkl.element_type, num_bits_per_copy=128)
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy, a_thread_layout, a_value_layout)
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mSFA_mkl.element_type,
                num_bits_per_copy=32)
            tidx_in_warpgroup = tidx % 128

            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2],
                     self.num_ab_stage),
                    stride=(
                        self.cta_tile_shape_mnk[2], 1,
                        self.cta_tile_shape_mnk[0]
                        * self.cta_tile_shape_mnk[2])))
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)), cutlass.Int32)
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)), cutlass.Boolean)
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)), cutlass.Boolean)

            # absorb_quant working set: f32 staging of 32 bf16 activations,
            # 4 packed e2m1 words, and the per-(thread, subrow) sSFA bases
            # (the same swizzled staging slots the cp.async sf path fills).
            yqf = cute.make_rmem_tensor(
                cute.make_layout((32,)), cutlass.Float32)
            qwords = cute.make_rmem_tensor(
                cute.make_layout((4,)), cutlass.Int32)
            sfa_quant_slices = []
            sA_w_base = None
            if cutlass.const_expr(self.absorb_quant):
                gs_x = norm_const_x_tensor[0]
                # fp4_quantize's output-scale recipe is rcp.approx-based
                # (quantization.cuh): osc = rcp(sfd * rcp(gs)); replicate it
                # exactly for bit-parity (exact division differs in the last
                # ulp and flips ~0.1% of e2m1 codes at rounding midpoints).
                rcp_gs_x = cute.arch.rcp_approx(gs_x)
                # i32-word view of the sA staging ring, taken at the
                # zero-offset base where every sub-byte recast convention
                # agrees (recasting a NON-zero fp4 slice iterator scatters:
                # phase-3 E1/E2 staging-split finding). Word offsets are
                # computed manually: stage*128*256/8 + row*256/8 + col/8.
                sA_w_base = cute.recast_ptr(sA.iterator,
                                            dtype=cutlass.Int32)
                for i in cutlass.range_constexpr(8):
                    sub_row = tidx_in_warpgroup // 8 + 16 * i
                    a_crd = 8 * ((sub_row // 8) % 4) + sub_row % 8
                    b_crd = sub_row // 32
                    sfa_quant_slices.append(
                        sSFA[((((a_crd, b_crd), None), None), None, None,
                              None)])

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(6, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # sSFA per-thread view (production swizzled staging slice).
            tAsSFA = sSFA[
                (
                    (
                        (
                            (
                                8 * (tidx_in_warpgroup // 32)
                                + (tidx_in_warpgroup % 8),
                                (tidx_in_warpgroup % 32) // 8,
                            ),
                            None,
                        ),
                        None,
                    ),
                    None,
                    None,
                    None,
                )
            ]

            while is_valid_tile:
                k_tile_cnt = fc1_k_tile_cnt
                if tile_info[5] == 1:
                    k_tile_cnt = fc2_k_tile_cnt

                # ---- per-row source mapping ----
                if tile_info[5] == 0:
                    # FC1: gather via token map, token = p2e[p] / topk
                    gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                    for i in range(8):
                        token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                        a_token_offset_tensor[i] = gToken_ml_tile[
                            token_ml_tile_offset]
                        a_predicate_tensor[i] = (
                            cutlass.Boolean(1)
                            if tile_info[0] * self.cta_tile_shape_mnk[0]
                            + token_ml_tile_offset < tile_info[4]
                            else cutlass.Boolean(0))
                        a_token_offset_tensor[i] = (
                            a_token_offset_tensor[i] // self.topk
                            if tile_info[0] * self.cta_tile_shape_mnk[0]
                            + token_ml_tile_offset < tile_info[4]
                            else 0)
                    token_ml_tile_offset = (
                        8 * (tidx_in_warpgroup // 32)
                        + 32 * ((tidx_in_warpgroup % 32) // 8)
                        + (tidx_in_warpgroup % 8))
                    sfa_row = gToken_ml_tile[token_ml_tile_offset] // self.topk
                    sfa_predicate_tensor[0] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset < tile_info[4]
                        else cutlass.Boolean(0))
                    if tile_info[0] * self.cta_tile_shape_mnk[0] \
                            + token_ml_tile_offset >= tile_info[4]:
                        sfa_row = cutlass.Int32(0)

                    tAgA = gA1_mkl[(None, None, 0, None, 0)]
                    tAgY = gY1_mkl[(None, None, 0, None, 0)]
                    A_gmem_thread_offset = cute.assume(
                        (tidx_in_warpgroup % 8) * 32, divby=32)
                    tAgSFA = gSFA1_mkl[(sfa_row, None, 0, None, 0)]

                    a_producer_state.reset_count()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(
                            a_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        a_pipeline.producer_acquire(
                            a_producer_state, peek_a_empty_status)
                        if cutlass.const_expr(self.absorb_quant):
                            # bf16 gather + inline NVFP4 quant into sA/sSFA.
                            tAgY_ktile = tAgY[
                                (None, None, a_producer_state.count)]
                            tAsA_ktile = tAsA_tiled[
                                (None, None, None, a_producer_state.index)]
                            for i in cutlass.range_constexpr(8):
                                if a_predicate_tensor[i]:
                                    y_off = (
                                        A_gmem_thread_offset + cute.assume(
                                            a_token_offset_tensor[i]
                                            * tAgY_ktile.layout[0].stride,
                                            divby=32))
                                    for cch in cutlass.range_constexpr(4):
                                        yv = cute.make_tensor(
                                            tAgY_ktile.iterator + y_off
                                            + 8 * cch,
                                            layout=cute.make_layout(
                                                (8,))).load().to(
                                                    cutlass.Float32)
                                        for ee in cutlass.range_constexpr(8):
                                            yqf[8 * cch + ee] = yv[ee]
                                    for grp in cutlass.range_constexpr(2):
                                        amax = fmax_f32(
                                            yqf[16 * grp],
                                            -yqf[16 * grp])
                                        for jj in cutlass.range_constexpr(
                                                1, 16):
                                            amax = fmax_f32(
                                                amax,
                                                fmax_f32(
                                                    yqf[16 * grp + jj],
                                                    -yqf[16 * grp + jj]))
                                        sfval = gs_x * (
                                            amax
                                            * cutlass.Float32(0.16666667))
                                        sfbyte = f32_to_e4m3_byte(sfval)
                                        sfd = e4m3_byte_to_f32(sfbyte)
                                        ind = fmin_f32(
                                            sfd * cutlass.Float32(512.0),
                                            cutlass.Float32(1.0))
                                        osc = ind * cute.arch.rcp_approx(
                                            fmax_f32(
                                                sfd,
                                                cutlass.Float32(
                                                    0.001953125))
                                            * rcp_gs_x)
                                        qwords[2 * grp] = e2m1x8_pack(
                                            yqf[16 * grp + 0] * osc,
                                            yqf[16 * grp + 1] * osc,
                                            yqf[16 * grp + 2] * osc,
                                            yqf[16 * grp + 3] * osc,
                                            yqf[16 * grp + 4] * osc,
                                            yqf[16 * grp + 5] * osc,
                                            yqf[16 * grp + 6] * osc,
                                            yqf[16 * grp + 7] * osc)
                                        qwords[2 * grp + 1] = e2m1x8_pack(
                                            yqf[16 * grp + 8] * osc,
                                            yqf[16 * grp + 9] * osc,
                                            yqf[16 * grp + 10] * osc,
                                            yqf[16 * grp + 11] * osc,
                                            yqf[16 * grp + 12] * osc,
                                            yqf[16 * grp + 13] * osc,
                                            yqf[16 * grp + 14] * osc,
                                            yqf[16 * grp + 15] * osc)
                                        gamma = 2 * (
                                            tidx_in_warpgroup % 8) + grp
                                        sfa_kt = sfa_quant_slices[i][
                                            (None, None, None, None,
                                             a_producer_state.index)]
                                        if cutlass.const_expr(
                                                self.absorb_dbg != 1):
                                            sf_dst = cute.make_tensor(
                                                cute.recast_ptr(
                                                    sfa_kt.iterator
                                                    + 512 * (gamma // 4)
                                                    + gamma % 4,
                                                    dtype=cutlass.Uint8),
                                                cute.make_layout((1,)))
                                            sf_dst[0] = sfbyte.to(
                                                cutlass.Uint8)
                                        if cutlass.const_expr(
                                                self.dbg_quant):
                                            mSfDbg[(
                                                a_token_offset_tensor[i],
                                                a_producer_state.count * 16
                                                + gamma)] = sfbyte
                                    if cutlass.const_expr(
                                            self.absorb_dbg != 2):
                                        word_off = (
                                            a_producer_state.index * 4096
                                            + (tidx_in_warpgroup // 8
                                               + 16 * i) * 32
                                            + (tidx_in_warpgroup % 8) * 4)
                                        qdst = cute.make_tensor(
                                            sA_w_base + word_off,
                                            cute.make_layout((4,)))
                                        cute.autovec_copy(qwords, qdst)
                                    if cutlass.const_expr(self.dbg_quant):
                                        for ww in cutlass.range_constexpr(4):
                                            mXqDbg[(
                                                a_token_offset_tensor[i],
                                                a_producer_state.count * 32
                                                + (tidx_in_warpgroup % 8)
                                                * 4 + ww)] = qwords[ww]
                            if cutlass.const_expr(self.absorb_dbg == 2):
                                # debug split: A bytes via the production
                                # cp.async path (quant only feeds sSFA)
                                tAgA_ktile = tAgA[
                                    (None, None, a_producer_state.count)]
                                for i in range(8):
                                    A_gmem_slice_offset = (
                                        A_gmem_thread_offset + cute.assume(
                                            a_token_offset_tensor[i]
                                            * tAgA_ktile.layout[0].stride,
                                            divby=32))
                                    A_gmem_slice_offset = cute.assume(
                                        A_gmem_slice_offset, divby=32)
                                    tAgA_slice = cute.make_tensor(
                                        tAgA_ktile.iterator
                                        + A_gmem_slice_offset,
                                        layout=cute.make_layout((32,)))
                                    tAsA_slice = cute.make_tensor(
                                        tAsA_ktile[(None, i, None)].iterator,
                                        layout=cute.make_layout((32,)))
                                    a_predicate_slice = cute.make_rmem_tensor(
                                        cute.make_layout((1,)),
                                        cutlass.Boolean)
                                    a_predicate_slice[0] = (
                                        a_predicate_tensor[i])
                                    cute.copy_atom_call(
                                        a_atom_copy, tAgA_slice, tAsA_slice,
                                        pred=a_predicate_slice)
                            if cutlass.const_expr(self.absorb_dbg == 1):
                                # debug split: sf bytes via the production
                                # cp.async path (quant only feeds sA)
                                tAgSFA_ktile = tAgSFA[
                                    (None, a_producer_state.count)]
                                tAsSFA_ktile = tAsSFA[
                                    (None, None, None, None,
                                     a_producer_state.index)]
                                for i in range(4):
                                    swizzled_iterator = (
                                        (tidx_in_warpgroup % 32) // 8 ^ i)
                                    tAgSFA_slice = cute.make_tensor(
                                        tAgSFA_ktile.iterator
                                        + 4 * swizzled_iterator,
                                        layout=cute.make_layout((4,)))
                                    tAsSFA_slice = cute.make_tensor(
                                        tAsSFA_ktile.iterator
                                        + 512 * swizzled_iterator,
                                        cute.make_layout((4,)))
                                    cute.copy_atom_call(
                                        sfa_atom_copy, tAgSFA_slice,
                                        tAsSFA_slice,
                                        pred=sfa_predicate_tensor)
                            cute.arch.fence_proxy(
                                "async.shared", space="cta")
                        else:
                            tAgA_ktile = tAgA[
                                (None, None, a_producer_state.count)]
                            tAsA_ktile = tAsA_tiled[
                                (None, None, None, a_producer_state.index)]
                            tAgSFA_ktile = tAgSFA[
                                (None, a_producer_state.count)]
                            tAsSFA_ktile = tAsSFA[
                                (None, None, None, None,
                                 a_producer_state.index)]
                            for i in range(8):
                                A_gmem_slice_offset = (
                                    A_gmem_thread_offset + cute.assume(
                                        a_token_offset_tensor[i]
                                        * tAgA_ktile.layout[0].stride,
                                        divby=32))
                                A_gmem_slice_offset = cute.assume(
                                    A_gmem_slice_offset, divby=32)
                                tAgA_slice = cute.make_tensor(
                                    tAgA_ktile.iterator + A_gmem_slice_offset,
                                    layout=cute.make_layout((32,)))
                                tAsA_slice = cute.make_tensor(
                                    tAsA_ktile[(None, i, None)].iterator,
                                    layout=cute.make_layout((32,)))
                                a_predicate_slice = cute.make_rmem_tensor(
                                    cute.make_layout((1,)), cutlass.Boolean)
                                a_predicate_slice[0] = a_predicate_tensor[i]
                                cute.copy_atom_call(
                                    a_atom_copy, tAgA_slice, tAsA_slice,
                                    pred=a_predicate_slice)
                            for i in range(4):
                                swizzled_iterator = (
                                    (tidx_in_warpgroup % 32) // 8 ^ i)
                                tAgSFA_slice = cute.make_tensor(
                                    tAgSFA_ktile.iterator
                                    + 4 * swizzled_iterator,
                                    layout=cute.make_layout((4,)))
                                tAsSFA_slice = cute.make_tensor(
                                    tAsSFA_ktile.iterator
                                    + 512 * swizzled_iterator,
                                    cute.make_layout((4,)))
                                cute.copy_atom_call(
                                    sfa_atom_copy, tAgSFA_slice, tAsSFA_slice,
                                    pred=sfa_predicate_tensor)
                        a_pipeline.producer_commit(a_producer_state)
                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if a_producer_state.count < k_tile_cnt:
                            peek_a_empty_status = (
                                a_pipeline.producer_try_acquire(
                                    a_producer_state))
                else:
                    # FC2: identity rows of c_q / c_sf (LINEAR)
                    for i in range(8):
                        token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                        row = (tile_info[0] * self.cta_tile_shape_mnk[0]
                               + token_ml_tile_offset)
                        a_predicate_tensor[i] = (
                            cutlass.Boolean(1) if row < tile_info[4]
                            else cutlass.Boolean(0))
                        a_token_offset_tensor[i] = (
                            row if row < tile_info[4] else 0)
                    token_ml_tile_offset = (
                        8 * (tidx_in_warpgroup // 32)
                        + 32 * ((tidx_in_warpgroup % 32) // 8)
                        + (tidx_in_warpgroup % 8))
                    sfa_row = (tile_info[0] * self.cta_tile_shape_mnk[0]
                               + token_ml_tile_offset)
                    sfa_predicate_tensor[0] = (
                        cutlass.Boolean(1) if sfa_row < tile_info[4]
                        else cutlass.Boolean(0))
                    if sfa_row >= tile_info[4]:
                        sfa_row = cutlass.Int32(0)

                    tAgA = gA2_mkl[(None, None, 0, None, 0)]
                    A_gmem_thread_offset = cute.assume(
                        (tidx_in_warpgroup % 8) * 32, divby=32)
                    tAgSFA = gSFA2_mkl[(sfa_row, None, 0, None, 0)]

                    a_producer_state.reset_count()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(
                            a_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        a_pipeline.producer_acquire(
                            a_producer_state, peek_a_empty_status)
                        tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                        tAsA_ktile = tAsA_tiled[
                            (None, None, None, a_producer_state.index)]
                        tAgSFA_ktile = tAgSFA[(None, a_producer_state.count)]
                        tAsSFA_ktile = tAsSFA[
                            (None, None, None, None, a_producer_state.index)]
                        for i in range(8):
                            A_gmem_slice_offset = (
                                A_gmem_thread_offset + cute.assume(
                                    a_token_offset_tensor[i]
                                    * tAgA_ktile.layout[0].stride, divby=32))
                            A_gmem_slice_offset = cute.assume(
                                A_gmem_slice_offset, divby=32)
                            tAgA_slice = cute.make_tensor(
                                tAgA_ktile.iterator + A_gmem_slice_offset,
                                layout=cute.make_layout((32,)))
                            tAsA_slice = cute.make_tensor(
                                tAsA_ktile[(None, i, None)].iterator,
                                layout=cute.make_layout((32,)))
                            a_predicate_slice = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean)
                            a_predicate_slice[0] = a_predicate_tensor[i]
                            cute.copy_atom_call(
                                a_atom_copy, tAgA_slice, tAsA_slice,
                                pred=a_predicate_slice)
                        for i in range(4):
                            swizzled_iterator = (
                                (tidx_in_warpgroup % 32) // 8 ^ i)
                            tAgSFA_slice = cute.make_tensor(
                                tAgSFA_ktile.iterator + 4 * swizzled_iterator,
                                layout=cute.make_layout((4,)))
                            tAsSFA_slice = cute.make_tensor(
                                tAsSFA_ktile.iterator + 512 * swizzled_iterator,
                                cute.make_layout((4,)))
                            cute.copy_atom_call(
                                sfa_atom_copy, tAgSFA_slice, tAsSFA_slice,
                                pred=sfa_predicate_tensor)
                        a_pipeline.producer_commit(a_producer_state)
                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if a_producer_state.count < k_tile_cnt:
                            peek_a_empty_status = (
                                a_pipeline.producer_try_acquire(
                                    a_producer_state))

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(6, unroll_full=True):
                    tile_info[idx] = sInfo[
                        (idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            a_pipeline.producer_tail(a_producer_state)

        # ==================================================================
        # TMA B/SFB warp (9): w13 for FC1 items, w2 for FC2 items
        # ==================================================================
        if warp_idx == self.tma_b_warp_id:
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(6, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                expert_idx = tile_info[2]
                slice_n = tile_info[1]
                k_tile_cnt = fc1_k_tile_cnt
                if tile_info[5] == 1:
                    k_tile_cnt = fc2_k_tile_cnt

                b_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = b_pipeline.producer_try_acquire(
                        b_producer_state)

                if tile_info[5] == 0:
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        b_pipeline.producer_acquire(
                            b_producer_state, peek_ab_empty_status)
                        tBsB_pipe = tBsB[(None, b_producer_state.index)]
                        tBsSFB_pipe = tBsSFB[(None, b_producer_state.index)]
                        tma_bar = b_pipeline.producer_get_barrier(
                            b_producer_state)
                        tBgB_slice = tBgB13[(None, slice_n, None, expert_idx)]
                        tBgSFB_slice = tBgSFB13[
                            (None, slice_n, None, expert_idx)]
                        cute.copy(
                            tma_atom_b13,
                            tBgB_slice[(None, b_producer_state.count)],
                            tBsB_pipe, tma_bar_ptr=tma_bar,
                            mcast_mask=b_full_mcast_mask)
                        cute.copy(
                            tma_atom_sfb13,
                            tBgSFB_slice[(None, b_producer_state.count)],
                            tBsSFB_pipe, tma_bar_ptr=tma_bar,
                            mcast_mask=sfb_full_mcast_mask)
                        b_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if b_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = (
                                b_pipeline.producer_try_acquire(
                                    b_producer_state))
                else:
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        b_pipeline.producer_acquire(
                            b_producer_state, peek_ab_empty_status)
                        tBsB_pipe = tBsB[(None, b_producer_state.index)]
                        tBsSFB_pipe = tBsSFB[(None, b_producer_state.index)]
                        tma_bar = b_pipeline.producer_get_barrier(
                            b_producer_state)
                        tBgB_slice = tBgB2[(None, slice_n, None, expert_idx)]
                        tBgSFB_slice = tBgSFB2[
                            (None, slice_n, None, expert_idx)]
                        cute.copy(
                            tma_atom_b2,
                            tBgB_slice[(None, b_producer_state.count)],
                            tBsB_pipe, tma_bar_ptr=tma_bar,
                            mcast_mask=b_full_mcast_mask)
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB_slice[(None, b_producer_state.count)],
                            tBsSFB_pipe, tma_bar_ptr=tma_bar,
                            mcast_mask=sfb_full_mcast_mask)
                        b_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if b_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = (
                                b_pipeline.producer_try_acquire(
                                    b_producer_state))

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(6, unroll_full=True):
                    tile_info[idx] = sInfo[
                        (idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            b_pipeline.producer_tail(b_producer_state)

        # ==================================================================
        # MMA warp (8): unified mainloop (k_tile_cnt is the only stage delta)
        # ==================================================================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols
                + self.num_sfa_tmem_cols, dtype=self.sf_dtype)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t) = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA))
            (tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t) = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB))

            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(6, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                k_tile_cnt = fc1_k_tile_cnt
                if tile_info[5] == 1:
                    k_tile_cnt = fc2_k_tile_cnt

                a_consumer_state.reset_count()
                peek_a_full_status = cutlass.Boolean(1)
                if a_consumer_state.count < k_tile_cnt:
                    peek_a_full_status = a_pipeline.consumer_try_wait(
                        a_consumer_state)
                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < k_tile_cnt:
                    peek_b_full_status = b_pipeline.consumer_try_wait(
                        b_consumer_state)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(k_tile_cnt):
                    a_pipeline.consumer_wait(
                        a_consumer_state, peek_a_full_status)
                    b_pipeline.consumer_wait(
                        b_consumer_state, peek_b_full_status)
                    s2t_stage_coord = (
                        None, None, None, None, b_consumer_state.index)
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_compact_s2t[s2t_stage_coord],
                        tCtSFA_compact_s2t)
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_compact_s2t[s2t_stage_coord],
                        tCtSFB_compact_s2t)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(
                            num_kblocks, unroll_full=True):
                        kblock_coord = (
                            None, None, kblock_idx, b_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(
                            tcgen05.Field.SFA,
                            tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(
                            tcgen05.Field.SFB,
                            tCtSFB[sf_kblock_coord].iterator)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[kblock_coord],
                            tCrB[kblock_coord], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    a_pipeline.consumer_release(a_consumer_state)
                    b_pipeline.consumer_release(b_consumer_state)
                    a_consumer_state.advance()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state)
                    b_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt:
                        peek_b_full_status = b_pipeline.consumer_try_wait(
                            b_consumer_state)

                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(6, unroll_full=True):
                    tile_info[idx] = sInfo[
                        (idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        # ==================================================================
        # Epilogue warps (0-3): FC1 swiglu+quant+TMA-store / FC2 finalize
        # ==================================================================
        if warp_idx <= self.epilog_warp_id[-1]:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx % 128
            # ---- FC1 partitions ----
            (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc_up, tTR_rAcc_gate) = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs))
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = (
                self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC))
            (tma_atom_c_l, bSG_sC, bSG_gC_partitioned) = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC))

            norm_const = norm_const_tensor[0]
            gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile, (None, None, None))
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
            tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
            tCrSFC = cute.make_rmem_tensor(
                tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype)
            tCrSFC_pvscale = cute.make_rmem_tensor_like(
                tCrSFC, cutlass.Float32)

            # ---- FC2 partitions ----
            (tiled_copy_t2r_f, tTR_tAcc_base_f, tTR_rAcc_f) = (
                self.fc2_epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgOut, fc2_epi_tile,
                    use_2cta_instrs))
            tTR_rC_f = cute.make_rmem_tensor(tTR_rAcc_f.shape, self.out_dtype)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)))
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(6, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            token_idx = cutlass.Int32(0)
            token_scale = cutlass.Float32(0.0)

            while is_valid_tile:
                expert_idx = tile_info[2]

                if tile_info[5] == 0:
                    # ====================== FC1 epilogue ======================
                    alpha_val = alpha1[expert_idx]
                    bSG_gC = bSG_gC_partitioned[
                        (None, None, None, tile_info[0], tile_info[1], 0)]
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_consumer_state.phase
                        reverse_subtile = (
                            cutlass.Boolean(True) if acc_stage_index == 0
                            else cutlass.Boolean(False))
                    else:
                        acc_stage_index = acc_consumer_state.index
                        reverse_subtile = cutlass.Boolean(False)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_stage_index)]
                    tCgSFC_mn = tCgSFC_mnl[(None, None, None, None, None, 0)]

                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3,
                                                cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                    for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                        real_subtile_idx = subtile_idx // 2
                        if cutlass.const_expr(self.overlapping_accum):
                            if reverse_subtile:
                                real_subtile_idx = (
                                    self.cta_tile_shape_mnk[1]
                                    // self.epi_tile_n_required
                                    - 1 - real_subtile_idx)
                        tTR_tAcc_mn_up = tTR_tAcc[
                            (None, None, None, real_subtile_idx * 2)]
                        tTR_tAcc_mn_gate = tTR_tAcc[
                            (None, None, None, real_subtile_idx * 2 + 1)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate,
                                  tTR_rAcc_gate)

                        if cutlass.const_expr(self.overlapping_accum):
                            if real_subtile_idx == \
                                    self.iter_acc_early_release_in_epilogue:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(
                                        acc_consumer_state)
                                acc_consumer_state.advance()

                        acc_vec_up = tTR_rAcc_up.load()
                        tCompute = cute.make_rmem_tensor(
                            acc_vec_up.shape, self.acc_dtype)
                        acc_vec_gate = tTR_rAcc_gate.load()
                        self._apply_swiglu_epilogue(
                            acc_vec_up, acc_vec_gate, alpha_val, tCompute)

                        # SFC generation + quant (production recipe).
                        sfc_subtile_idx_mn = (
                            tile_info[0] * self.epi_tile_cnt[0],
                            tile_info[1] * self.epi_tile_cnt[1]
                            + real_subtile_idx)
                        tCgSFC = tCgSFC_mn[
                            (None, None, None, *sfc_subtile_idx_mn)]
                        tTR_rAcc_frg = cute.logical_divide(
                            tCompute, cute.make_layout(self.sf_vec_size))
                        acc_frg = tTR_rAcc_frg.load()
                        abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                        abs_acc_frg = type(acc_frg)(
                            abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
                        for vi in cutlass.range_constexpr(
                                abs_acc_frg.shape[1]):
                            tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
                        for vi in cutlass.range_constexpr(
                                0, abs_acc_frg.shape[1], 2):
                            tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrSFC_pvscale[vi],
                                     tCrSFC_pvscale[vi + 1]),
                                    (self.c_dtype_rcp_limit,
                                     self.c_dtype_rcp_limit)))
                            tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrSFC_pvscale[vi],
                                     tCrSFC_pvscale[vi + 1]),
                                    (norm_const, norm_const)))
                        tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))
                        cute.autovec_copy(tCrSFC, tCgSFC)

                        tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                        fp32_max = cutlass.Float32(3.40282346638528859812e38)
                        for vi in cutlass.range_constexpr(
                                0, cute.size(tCrSFC), 2):
                            acc_scale = cute.arch.mul_packed_f32x2(
                                (cute.arch.rcp_approx(
                                    tCrSFC_qpvscale_up[vi]),
                                 cute.arch.rcp_approx(
                                     tCrSFC_qpvscale_up[vi + 1])),
                                (norm_const, norm_const))
                            acc_scale_min0 = fmin(
                                acc_scale[0], fp32_max, nan=True)
                            acc_scale_min1 = fmin(
                                acc_scale[1], fp32_max, nan=True)
                            vec0 = tTR_rAcc_frg[None, vi]
                            vec1 = tTR_rAcc_frg[None, vi + 1]
                            for ei in cutlass.range_constexpr(
                                    self.sf_vec_size):
                                vec0[ei], vec1[ei] = (
                                    cute.arch.mul_packed_f32x2(
                                        (vec0[ei], vec1[ei]),
                                        (acc_scale_min0, acc_scale_min1)))

                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s, tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c_l, bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)])
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(not self.overlapping_accum):
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                    # publish: all TMA stores of this item complete, then
                    # bump the tile's done counter at gpu scope.
                    if cutlass.const_expr(not self.skip_release):
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cp_async_bulk_wait_group_0()
                            with cute.arch.elect_one():
                                red_release_add_u32(
                                    elem_ptr(mDone, (tile_info[0],)),
                                    cutlass.Int32(1))
                else:
                    # ====================== FC2 finalize ======================
                    alpha_val2 = alpha2[expert_idx]
                    tile_m_start = (tile_info[0] * self.cta_tile_shape_mnk[0])
                    permuted_row = tile_m_start + epi_tidx
                    expanded_idx = token_id_mapping_tensor[permuted_row]
                    is_valid_row = permuted_row < tile_info[4]

                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_consumer_state.phase
                        reverse_subtile = (
                            cutlass.Boolean(True) if acc_stage_index == 0
                            else cutlass.Boolean(False))
                    else:
                        acc_stage_index = acc_consumer_state.index
                        reverse_subtile = cutlass.Boolean(False)
                    tTR_tAcc = tTR_tAcc_base_f[
                        (None, None, None, None, None, acc_stage_index)]

                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3,
                                                cute.rank(tTR_tAcc))
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                    if is_valid_row:
                        token_idx = expanded_idx // self.topk
                        topk_idx = expanded_idx % self.topk
                        token_scale = token_final_scales[
                            (token_idx, topk_idx)]
                        alpha_val2 = alpha_val2 * token_scale

                    for subtile_idx in cutlass.range(subtile_cnt):
                        real_subtile_idx = subtile_idx
                        if cutlass.const_expr(self.overlapping_accum):
                            if reverse_subtile:
                                real_subtile_idx = (
                                    subtile_cnt - 1 - subtile_idx)
                        tTR_tAcc_mn = tTR_tAcc[
                            (None, None, None, real_subtile_idx)]
                        cute.copy(tiled_copy_t2r_f, tTR_tAcc_mn, tTR_rAcc_f)

                        if cutlass.const_expr(self.overlapping_accum):
                            if subtile_idx == self.fc2_iter_acc_early_release:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(
                                        acc_consumer_state)
                                acc_consumer_state.advance()

                        acc_vec = tTR_rAcc_f.load()
                        acc_vec_final = alpha_val2 * acc_vec
                        tTR_rC_f.store(acc_vec_final.to(self.out_dtype))
                        if is_valid_row:
                            rOut_epi = cute.make_tensor(
                                tTR_rC_f.iterator, fc2_epi_layout)
                            base_coord_n = (
                                tile_info[1] * self.cta_tile_shape_mnk[1]
                                + real_subtile_idx * cute.size(tTR_rC_f))
                            scatter_out = cute.domain_offset(
                                (token_idx, 0, 0), mOut)
                            for index in cutlass.range(
                                    self.fc2_epi_loop_size, unroll_full=True):
                                coord_n = (base_coord_n
                                           + index * self.fc2_element_offset)
                                scatter_out_offset = cute.domain_offset(
                                    (0, coord_n, 0), scatter_out)
                                rOut_epi_packed = rOut_epi[index, None, None]
                                vectorized_atomic_add_bf16x8(
                                    rOut_epi_packed, scatter_out_offset)

                    if cutlass.const_expr(not self.overlapping_accum):
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(6, unroll_full=True):
                    tile_info[idx] = sInfo[
                        (idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            # In-kernel self-reset: the last CTA out re-zeroes the done
            # counters + pop cursor + exit slot, so the next invocation
            # needs NO host fill node (the buffer is zeroed once at alloc).
            # Ordering: this CTA's done[] red.release publishes precede the
            # epilog barrier above; the acq_rel fetch-add then release-
            # orders them for other CTAs and acquire-orders every other
            # CTA's publishes for the winner, whose plain re-zero stores
            # are ordered before the next invocation by kernel completion.
            if warp_idx == self.epilog_warp_id[0]:
                if tidx % 32 == 0:
                    exit_old = atom_acq_rel_add_ret_u32(
                        elem_ptr(mDone, (self.n_tiles_bound + 1,)),
                        cutlass.Int32(1))
                    if exit_old == gdim - 1:
                        for ri in cutlass.range(self.n_tiles_bound + 2):
                            mDone[ri] = cutlass.Int32(0)
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------
    # helpers shared with the production sources
    # ------------------------------------------------------------------
    @cute.jit
    def _apply_swiglu_epilogue(self, acc_vec_up, acc_vec_gate, alpha_val,
                               tCompute):
        LOG2_E = cutlass.Float32(1.4426950408889634)
        for i in cutlass.range_constexpr(0, cute.size(acc_vec_up.shape), 2):
            acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                (acc_vec_up[i], acc_vec_up[i + 1]),
                (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)))
            acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                (acc_vec_gate[i], acc_vec_gate[i + 1]),
                (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)))
            tCompute_log2e = cute.arch.mul_packed_f32x2(
                (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                (-LOG2_E, -LOG2_E))
            (tCompute[i], tCompute[i + 1]) = cute.arch.add_packed_f32x2(
                (cute.math.exp2(tCompute_log2e[0], fastmath=True),
                 cute.math.exp2(tCompute_log2e[1], fastmath=True)),
                (1.0, 1.0))
            tCompute[i] = cute.arch.rcp_approx(tCompute[i])
            tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
            (tCompute[i], tCompute[i + 1]) = cute.arch.mul_packed_f32x2(
                (tCompute[i], tCompute[i + 1]),
                (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]))
            (tCompute[i], tCompute[i + 1]) = cute.arch.mul_packed_f32x2(
                (tCompute[i], tCompute[i + 1]),
                (acc_vec_up_alpha[0], acc_vec_up_alpha[1]))

    @property
    def c_dtype_rcp_limit(self) -> float:
        return 1.0 / 6.0  # Float4E2M1FN max magnitude

    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gC_mnl, epi_tile,
                                       use_2cta_instrs):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype,
            self.acc_dtype, epi_tile, use_2cta_instrs)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def fc2_epilog_tmem_copy_and_partition(self, tidx, tAcc, gC_mnl, epi_tile,
                                           use_2cta_instrs):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, utils.LayoutEnum.ROW_MAJOR,
            self.out_dtype, self.acc_dtype, epi_tile, use_2cta_instrs)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx,
                                       sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(self, tidx, atom, gC_mnl, epi_tile,
                                       sC):
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            sC_for_tma_partition, gC_for_tma_partition)
        return tma_atom_c, bSG_sC, bSG_gC


def mSFC_linear_view(mSFC_mnl: cute.Tensor, mCq: cute.Tensor,
                     sf_vec_size: int) -> cute.Tensor:
    """LINEAR [perm_m, I/16] view of the intermediate sf for the FC2 LDGSTS
    path (rank-3 (m, k16, 1), row-major)."""
    k16 = mCq.shape[1] // sf_vec_size
    return cute.make_tensor(
        mSFC_mnl.iterator,
        cute.make_layout((mCq.shape[0], k16, 1), stride=(k16, 1, 0)))


class MegaPersistentMoE(MegaPersistentMoEKernel):
    """Pointer-level entry (production op idiom): builds the cute tensors from
    raw device pointers + the fixed shapes, then runs the megakernel."""

    def __init__(self, h, inter, num_experts, topk, grid, n_tokens,
                 perm_m, n_tiles, mma_n=128, on_the_fly=False, dyn_pop=False,
                 absorb_quant=False):
        super().__init__(h, inter, num_experts, topk, grid, mma_n=mma_n,
                         on_the_fly=on_the_fly, dyn_pop=dyn_pop,
                         absorb_quant=absorb_quant)
        self.n_tokens = n_tokens
        self.perm_m = perm_m
        self.n_tiles = n_tiles
        # cursor slot index inside the done buffer ([n_tiles + 2] i32)
        self.n_tiles_bound = n_tiles

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,        # x_q fp4
        sfa_ptr: cute.Pointer,      # x_sf e4m3 LINEAR
        b13_ptr: cute.Pointer,      # w13 fp4
        sfb13_ptr: cute.Pointer,    # w13 sf e4m3 (atom flat)
        b2_ptr: cute.Pointer,       # w2 fp4
        sfb2_ptr: cute.Pointer,     # w2 sf e4m3 (atom flat)
        c_ptr: cute.Pointer,        # c_q fp4 (intermediate)
        sfc_ptr: cute.Pointer,      # c_sf e4m3 LINEAR (intermediate)
        norm_ptr: cute.Pointer,     # gs_c f32 [1]
        out_ptr: cute.Pointer,      # out bf16 [ntok, H]
        p2e_ptr: cute.Pointer,      # i32 [perm_m]
        tfs_ptr: cute.Pointer,      # f32 [ntok, topk]
        alpha1_ptr: cute.Pointer,   # f32 [E]
        alpha2_ptr: cute.Pointer,   # f32 [E]
        items_ptr: cute.Pointer,    # i32 [n_items, 6]
        done_ptr: cute.Pointer,     # i32 [n_tiles + 2] (counters + cursor)
        n_items: cutlass.Int32,
        t2e_ptr: cute.Pointer,      # i32 [n_tiles] (on_the_fly)
        mn_lim_ptr: cute.Pointer,   # i32 [n_tiles] cumulative (on_the_fly)
        nnet_ptr: cute.Pointer,     # i32 [1] (on_the_fly)
        y_ptr: cute.Pointer,        # bf16 [ntok, H] (absorb_quant)
        norm_x_ptr: cute.Pointer,   # gs_x f32 [1] (absorb_quant)
        xq_dbg_ptr: cute.Pointer,   # i32 [ntok, H/8] (dbg_quant)
        sf_dbg_ptr: cute.Pointer,   # i32 [ntok, H/16] (dbg_quant)
        n_tokens: cutlass.Int32,    # dynamic: out rows need not divide 128
        stream: cuda.CUstream,
    ):
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout(
                (n_tokens, self.H, 1), order=(1, 0, 2)))
        sfa = cute.make_tensor(
            sfa_ptr, cute.make_ordered_layout(
                (n_tokens, self.H // self.sf_vec_size, 1),
                order=(1, 0, 2)))
        b13 = cute.make_tensor(
            b13_ptr, cute.make_ordered_layout(
                (2 * self.I, self.H, self.E), order=(1, 0, 2)))
        sfb13 = cute.make_tensor(
            sfb13_ptr, cute.make_layout(
                (self.E * 2 * self.I * (self.H // self.sf_vec_size),)))
        b2 = cute.make_tensor(
            b2_ptr, cute.make_ordered_layout(
                (self.H, self.I, self.E), order=(1, 0, 2)))
        sfb2 = cute.make_tensor(
            sfb2_ptr, cute.make_layout(
                (self.E * self.H * (self.I // self.sf_vec_size),)))
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout(
                (self.perm_m, self.I, 1), order=(1, 0, 2)))
        sfc = cute.make_tensor(
            sfc_ptr, cute.make_layout(
                (self.perm_m * (self.I // self.sf_vec_size),)))
        norm_const = cute.make_tensor(norm_ptr, cute.make_layout((1,)))
        out = cute.make_tensor(
            out_ptr, cute.make_ordered_layout(
                (n_tokens, self.H, 1), order=(1, 0, 2)))
        p2e = cute.make_tensor(p2e_ptr, cute.make_layout((self.perm_m,)))
        tfs = cute.make_tensor(
            tfs_ptr, cute.make_ordered_layout(
                (n_tokens, self.topk), order=(1, 0)))
        alpha1 = cute.make_tensor(alpha1_ptr, cute.make_layout((self.E,)))
        alpha2 = cute.make_tensor(alpha2_ptr, cute.make_layout((self.E,)))
        items = cute.make_tensor(
            items_ptr, cute.make_ordered_layout((n_items, 6), order=(1, 0)))
        done = cute.make_tensor(done_ptr,
                                cute.make_layout((self.n_tiles + 2,)))
        t2e = cute.make_tensor(t2e_ptr, cute.make_layout((self.n_tiles,)))
        mn_lim = cute.make_tensor(mn_lim_ptr,
                                  cute.make_layout((self.n_tiles,)))
        nnet = cute.make_tensor(nnet_ptr, cute.make_layout((1,)))
        y = cute.make_tensor(
            y_ptr, cute.make_ordered_layout(
                (n_tokens, self.H, 1), order=(1, 0, 2)))
        norm_x = cute.make_tensor(norm_x_ptr, cute.make_layout((1,)))
        xq_dbg = cute.make_tensor(
            xq_dbg_ptr, cute.make_ordered_layout(
                (n_tokens, self.H // 8), order=(1, 0)))
        sf_dbg = cute.make_tensor(
            sf_dbg_ptr, cute.make_ordered_layout(
                (n_tokens, self.H // 16), order=(1, 0)))

        self(a, sfa, b13, sfb13, b2, sfb2, c, sfc, norm_const, out, p2e,
             tfs, alpha1, alpha2, items, done, n_items, t2e, mn_lim, nnet,
             y, norm_x, xq_dbg, sf_dbg, stream)


# ---------------------------------------------------------------------------
# Production wrapper: compile cache + persistent per-bucket state, callable
# from fused_moe_cute_dsl.run_moe_nvfp4_impl behind
# TRTLLM_OPTRT_MOE_MEGAKERNEL_V2 (opt-in; default off).
# ---------------------------------------------------------------------------
import os as _os

import torch


def megakernel_v2_enabled() -> bool:
    """Opt-in gate for the persistent decode-MoE megakernel (phase 3).

    ``TRTLLM_OPTRT_MOE_MEGAKERNEL_V2`` truthy -> the moe_sort-driven
    FC1+FC2 chain is executed by ONE persistent grid with the on-device
    work-item producer and atomic-cursor scheduling. Requires
    tile_size=128 (decode_1cta) and the fused-finalize path.
    """
    val = _os.environ.get("TRTLLM_OPTRT_MOE_MEGAKERNEL_V2", "0")
    return val.strip().lower() not in ("", "0", "off", "false", "no")


class _MegaV2Instance:
    """One compiled megakernel + its persistent buffers for a fixed
    (device, max_tiles, perm_bound, E, topk, H, I) bucket."""

    def __init__(self, device: torch.device, h: int, inter: int,
                 num_experts: int, top_k: int, max_tiles: int,
                 perm_bound: int, sample_args: tuple):
        import cuda.bindings.driver as cuda_driver
        import cutlass
        import cutlass.cute as cute

        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
        self.donecur = torch.zeros(max_tiles + 2, dtype=torch.int32,
                                   device=device)
        self.c_q = torch.empty(perm_bound, inter // 2, dtype=torch.uint8,
                               device=device)
        self.c_sf = torch.empty(perm_bound, inter // 16, dtype=torch.uint8,
                                device=device)
        # static-list inputs unused in on_the_fly mode; 1-element dummies
        self.items_dummy = torch.zeros(1, 6, dtype=torch.int32, device=device)
        kern = MegaPersistentMoE(
            h, inter, num_experts, top_k, n_sm, 0, perm_bound, max_tiles,
            mma_n=128, on_the_fly=True, dyn_pop=True, absorb_quant=False)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled = cute.compile(kern.wrapper, *sample_args, stream,
                                     options="--opt-level 2")


_V2_CACHE = {}


def _ptr(dtype_name, t, align):
    import cutlass
    import cutlass.cute as cute

    from ..utils import make_ptr
    return make_ptr(getattr(cutlass, dtype_name), t.data_ptr(),
                    cute.AddressSpace.gmem, assumed_align=align)


def run_mega_persistent_moe_v2(
    x_q: torch.Tensor,            # fp4-packed [ntok, H/2] (uint8/fp4x2 view)
    x_sf: torch.Tensor,           # e4m3 LINEAR [ntok, H/16] (uint8 view)
    w13: torch.Tensor,            # fp4-packed [E, 2I, H/2]
    w13_sf: torch.Tensor,         # e4m3 atom-flat [E, ...]
    w2: torch.Tensor,             # fp4-packed [E, H, I/2]
    w2_sf: torch.Tensor,          # e4m3 atom-flat [E, ...]
    alpha1: torch.Tensor,         # f32 [E] (fc1_global_scale)
    alpha2: torch.Tensor,         # f32 [E] (fc2_global_scale)
    fc2_input_scale: torch.Tensor,  # f32 [1] (the FC1-epilogue requant gs)
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,   # f32 [ntok, topk]
    moe_output: torch.Tensor,    # bf16 [ntok, H]; rows pre-zeroed by caller
    *,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Persistent-grid FC1+FC2 for the decode_1cta fused-finalize path.

    Numerics: identical kernels/recipe to
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell +
    cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell (phase-2 parity:
    cos 0.99999 between the chains, both at the requant floor vs true f32).
    Graph-replay-safe: work items are derived on device from the moe_sort
    outputs each invocation; the only host-visible state is the
    done-counter/cursor buffer zeroed by one fill node per call.
    """
    import cuda.bindings.driver as cuda_driver
    import cutlass

    device = moe_output.device
    ntok = moe_output.size(0)
    max_tiles = tile_idx_to_expert_idx.numel()
    perm_bound = permuted_idx_to_expanded_idx.numel()
    key = (device.index, max_tiles, perm_bound, num_local_experts, top_k,
           hidden_size, intermediate_size)

    inst = _V2_CACHE.get(key)

    def build_args(instance):
        return (
            _ptr("Float4E2M1FN", x_q, 32), _ptr("Float8E4M3FN", x_sf, 16),
            _ptr("Float4E2M1FN", w13, 32), _ptr("Float8E4M3FN", w13_sf, 16),
            _ptr("Float4E2M1FN", w2, 32), _ptr("Float8E4M3FN", w2_sf, 16),
            _ptr("Float4E2M1FN", instance.c_q, 32),
            _ptr("Float8E4M3FN", instance.c_sf, 16),
            _ptr("Float32", fc2_input_scale, 16),
            _ptr("BFloat16", moe_output, 16),
            _ptr("Int32", permuted_idx_to_expanded_idx, 16),
            _ptr("Float32", token_final_scales, 16),
            _ptr("Float32", alpha1, 16), _ptr("Float32", alpha2, 16),
            _ptr("Int32", instance.items_dummy, 16),
            _ptr("Int32", instance.donecur, 16),
            cutlass.Int32(0),
            _ptr("Int32", tile_idx_to_expert_idx, 16),
            _ptr("Int32", tile_idx_to_mn_limit, 16),
            _ptr("Int32", num_non_exiting_tiles, 16),
            _ptr("BFloat16", moe_output, 16),  # y unused (absorb off)
            _ptr("Float32", fc2_input_scale, 16),  # gs_x unused (absorb off)
            # xq_dbg / sf_dbg: only dereferenced under dbg_quant (off here), but
            # the wrapper still builds a cute.Tensor from each pointer, so they
            # must be present and address-valid. Reuse moe_output's storage
            # (never read): its byte span [ntok*H*2] covers the i32 [ntok, H/8]
            # and [ntok, H/16] views the wrapper constructs.
            _ptr("Int32", moe_output, 16),
            _ptr("Int32", moe_output, 16),
            cutlass.Int32(ntok),
        )

    if inst is None:
        probe = _MegaV2Instance.__new__(_MegaV2Instance)
        probe.donecur = torch.zeros(max_tiles + 2, dtype=torch.int32,
                                    device=device)
        probe.c_q = torch.empty(perm_bound, intermediate_size // 2,
                                dtype=torch.uint8, device=device)
        probe.c_sf = torch.empty(perm_bound, intermediate_size // 16,
                                 dtype=torch.uint8, device=device)
        probe.items_dummy = torch.zeros(1, 6, dtype=torch.int32,
                                        device=device)
        inst = _MegaV2Instance(device, hidden_size, intermediate_size,
                               num_local_experts, top_k, max_tiles,
                               perm_bound, build_args(probe))
        inst.donecur = probe.donecur
        inst.c_q = probe.c_q
        inst.c_sf = probe.c_sf
        inst.items_dummy = probe.items_dummy
        _V2_CACHE[key] = inst

    # No per-call reset: the kernel self-resets the done/cursor/exit buffer
    # (the last CTA out re-zeroes it), so the only zeroing is the one-time
    # torch.zeros at instance alloc. Saves a fill node per MoE layer.
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    inst.compiled(*build_args(inst), stream)
    return moe_output

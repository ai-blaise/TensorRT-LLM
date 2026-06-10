# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL port of the fused low-rank gate (REAP gated-norm) decode kernel.

Computes y = x * sigmoid(silu(x @ Wd.T) @ Wu) for rank-R gates in a single
kernel launch; the Triton implementation in fused_lowrank_gate.py needs a
split-K pair. One CTA cluster handles one row of x: each CTA reduces a
K-slice of the rank-R down-projection, the fp32 partials cross the cluster
through distributed shared memory (st.async + mbarrier), and every CTA then
applies silu -> up-proj -> sigmoid -> mul on its N-slice. All traffic is
128-bit vectorized FMA work; at rank 16 tensor cores lose to plain FMA.

Wd is read as bf16. The eager chain upcasts the bf16 master weight to fp32
before its SGEMM, and bf16 -> fp32 conversion is exact, so an in-register
upcast of the bf16 weight produces bit-identical products at half the bytes.

Rounding points mirror the eager chain and the Triton kernel: fp32 down-proj
accumulate, silu -> bf16, fp32 up-proj accumulate -> bf16, sigmoid -> bf16,
final mul in fp32 -> bf16.
"""

import os

import torch

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

CUTE_LOWRANK_GATE_AVAILABLE = IS_CUTLASS_DSL_AVAILABLE

_VEC = 8  # bf16 elements per 128-bit load.

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Float32, Int32
    from cutlass.cutlass_dsl import T, dsl_user_op

    class _CUDAGraphCompatibleWrapper:
        """DLPack exporter that stays valid under CUDA graph capture."""

        def __init__(self, tensor):
            self._tensor = tensor

        def __dlpack__(self, stream=None):
            return self._tensor.__dlpack__(stream=-1)

        def __dlpack_device__(self):
            return self._tensor.__dlpack_device__()

    @dsl_user_op
    def _elem_pointer(x: cute.Tensor,
                      coord: cute.Coord,
                      *,
                      loc=None,
                      ip=None) -> cute.Pointer:
        return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

    @dsl_user_op
    def _peer_smem_addr(smem_ptr: cute.Pointer,
                        peer_cta_rank: cutlass.Int32,
                        *,
                        loc=None,
                        ip=None) -> cutlass.Int32:
        smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [smem_ptr_i32,
                 Int32(peer_cta_rank).ir_value()],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            ))

    @dsl_user_op
    def _store_remote_f32(val: Float32,
                          smem_ptr: cute.Pointer,
                          mbar_ptr: cute.Pointer,
                          peer_cta_rank: cutlass.Int32,
                          *,
                          loc=None,
                          ip=None) -> None:
        remote_smem = _peer_smem_addr(smem_ptr, peer_cta_rank, loc=loc,
                                      ip=ip).ir_value()
        remote_mbar = _peer_smem_addr(mbar_ptr, peer_cta_rank, loc=loc,
                                      ip=ip).ir_value()
        llvm.inline_asm(
            None,
            [remote_smem,
             Float32(val).ir_value(loc=loc, ip=ip), remote_mbar],
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 "
            "[$0], $1, [$2];",
            "r,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    class _LowRankGateKernel:
        """One-launch fused gate: grid (cluster_n, M), one cluster per row."""

        def __init__(self, n: int, r: int, cluster_n: int):
            assert n % _VEC == 0
            vecs = n // _VEC
            assert vecs % cluster_n == 0
            vecs_per_cta = vecs // cluster_n
            threads = next(
                (t for t in range(256, 31, -32) if vecs_per_cta % t == 0),
                None)
            assert threads is not None, f"no thread count divides {vecs_per_cta}"
            self.n = n
            self.r = r
            self.cluster_n = cluster_n
            self.threads = threads
            self.iters = vecs_per_cta // threads
            self.warps = threads // 32

        def _smem_bytes(self) -> int:
            size = self.warps * self.r * 4 + self.r * 4
            if self.cluster_n > 1:
                size += self.cluster_n * self.r * 4 + 8
            return size

        @cute.jit
        def __call__(self, mX: cute.Tensor, mWd: cute.Tensor,
                     mWu: cute.Tensor, mY: cute.Tensor,
                     stream: cuda_driver.CUstream):
            gX = cute.zipped_divide(mX, (1, _VEC))
            gWd = cute.zipped_divide(mWd, (1, _VEC))
            gWu = cute.zipped_divide(mWu, (1, _VEC))
            gY = cute.zipped_divide(mY, (1, _VEC))
            self.kernel(gX, gWd, gWu, gY).launch(
                grid=[self.cluster_n, mX.shape[0], 1],
                block=[self.threads, 1, 1],
                cluster=([self.cluster_n, 1, 1]
                         if cutlass.const_expr(self.cluster_n > 1) else None),
                smem=self._smem_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, gX: cute.Tensor, gWd: cute.Tensor, gWu: cute.Tensor,
                   gY: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            ks, row, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32

            smem = cutlass.utils.SmemAllocator()
            s_warp = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((self.warps, self.r), stride=(self.r, 1)),
                byte_alignment=8)
            s_gate = smem.allocate_tensor(cutlass.Float32,
                                          cute.make_layout(self.r),
                                          byte_alignment=8)
            if cutlass.const_expr(self.cluster_n > 1):
                s_part = smem.allocate_tensor(
                    cutlass.Float32,
                    cute.make_layout((self.cluster_n, self.r),
                                     stride=(self.r, 1)),
                    byte_alignment=8)
                mbar = smem.allocate_array(cutlass.Int64, num_elems=1)
                if tidx == 0:
                    cute.arch.mbarrier_init(mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.cluster_arrive_relaxed()

            base = ks * (self.iters * self.threads)

            # Phase A: rank-R down-projection over this CTA's K-slice.
            acc = cute.make_fragment(cute.make_layout(self.r),
                                     cutlass.Float32)
            acc.fill(0.0)
            for i in cutlass.range_constexpr(self.iters):
                vk = base + i * self.threads + tidx
                xv = gX[(None, (row, vk))].load().to(cutlass.Float32)
                for r in cutlass.range_constexpr(self.r):
                    wv = gWd[(None, (r, vk))].load().to(cutlass.Float32)
                    acc[r] = acc[r] + (xv * wv).reduce(
                        cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)

            for r in cutlass.range_constexpr(self.r):
                v = acc[r]
                for step in cutlass.range_constexpr(5):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << step)
                acc[r] = v

            if lane == 0:
                for r in cutlass.range_constexpr(self.r):
                    s_warp[warp, r] = acc[r]
            cute.arch.barrier()

            # Reduce warp partials (plus cluster partials), then
            # silu -> bf16 -> fp32, shared by every thread in phase B.
            if cutlass.const_expr(self.cluster_n > 1):
                cute.arch.cluster_wait()
                if warp == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar, self.cluster_n * self.r * 4)
                if tidx < self.r:
                    part = s_warp[0, tidx]
                    for w in cutlass.range_constexpr(1, self.warps):
                        part = part + s_warp[w, tidx]
                    for peer in cutlass.range_constexpr(self.cluster_n):
                        _store_remote_f32(part,
                                          _elem_pointer(s_part, (ks, tidx)),
                                          mbar, Int32(peer))
                cute.arch.mbarrier_wait(mbar, phase=0)
                if tidx < self.r:
                    g = s_part[0, tidx]
                    for c in cutlass.range_constexpr(1, self.cluster_n):
                        g = g + s_part[c, tidx]
                    g = g * (Float32(1.0) /
                             (Float32(1.0) + cute.math.exp(-g)))
                    s_gate[tidx] = g.to(cutlass.BFloat16).to(cutlass.Float32)
            else:
                if tidx < self.r:
                    g = s_warp[0, tidx]
                    for w in cutlass.range_constexpr(1, self.warps):
                        g = g + s_warp[w, tidx]
                    g = g * (Float32(1.0) /
                             (Float32(1.0) + cute.math.exp(-g)))
                    s_gate[tidx] = g.to(cutlass.BFloat16).to(cutlass.Float32)
            cute.arch.barrier()

            # Phase B: up-proj -> bf16 -> sigmoid -> bf16 -> x*gate -> bf16.
            for i in cutlass.range_constexpr(self.iters):
                vn = base + i * self.threads + tidx
                dot = s_gate[0] * gWu[(None, (0, vn))].load().to(
                    cutlass.Float32)
                for r in cutlass.range_constexpr(1, self.r):
                    dot = dot + s_gate[r] * gWu[(None, (r, vn))].load().to(
                        cutlass.Float32)
                dot = dot.to(cutlass.BFloat16).to(cutlass.Float32)
                ones = cute.full_like(dot, 1.0)
                gate = ones / (ones + cute.math.exp(-dot))
                gate = gate.to(cutlass.BFloat16).to(cutlass.Float32)
                xv = gX[(None, (row, vn))].load().to(cutlass.Float32)
                gY[(None, (row, vn))] = (xv * gate).to(cutlass.BFloat16)

    _compile_cache = {}

    def _to_cute_2d_dynamic_rows(t: torch.Tensor):
        return from_dlpack(_CUDAGraphCompatibleWrapper(t.detach()),
                           assumed_align=16).mark_compact_shape_dynamic(
                               mode=0, stride_order=(0, 1))

    def _to_cute_2d_static(t: torch.Tensor):
        return from_dlpack(_CUDAGraphCompatibleWrapper(t.detach()),
                           assumed_align=16)

    def _default_cluster_n() -> int:
        # B200 graph-replay sweep (M=4/16): CN7 3.95/4.19us, CN4 4.26/4.34us,
        # CN2 6.50/6.54us, CN1 10.57/10.76us vs Triton pair 6.18/6.72us.
        return int(
            os.environ.get("TRTLLM_OPTRT_LOWRANK_GATE_CUTE_CLUSTER", "7"))

    @torch.library.custom_op("trtllm::cute_lowrank_gate", mutates_args=())
    def cute_lowrank_gate(x: torch.Tensor, wd_bf16: torch.Tensor,
                          wu_t_bf16: torch.Tensor) -> torch.Tensor:
        """y = x * sigmoid(silu(x @ wd_bf16.T) @ wu_t_bf16) for rank-R gates.

        x: [M, N] bf16 contiguous. wd_bf16: [R, N] bf16 contiguous
        (gate_down.weight). wu_t_bf16: [R, N] bf16 contiguous
        (gate_up.weight.T).
        """
        n = x.shape[1]
        r = wd_bf16.shape[0]
        y = torch.empty_like(x)
        cluster_n = _default_cluster_n()
        key = (n, r, cluster_n)
        x_t = _to_cute_2d_dynamic_rows(x)
        y_t = _to_cute_2d_dynamic_rows(y)
        wd_t = _to_cute_2d_static(wd_bf16)
        wu_t = _to_cute_2d_static(wu_t_bf16)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _compile_cache.get(key)
        if compiled is None:
            kernel = _LowRankGateKernel(n, r, cluster_n)
            compiled = cute.compile(kernel, x_t, wd_t, wu_t, y_t, stream)
            _compile_cache[key] = compiled
        compiled(x_t, wd_t, wu_t, y_t, stream)
        return y

    @cute_lowrank_gate.register_fake
    def _(x, wd_bf16, wu_t_bf16):
        return torch.empty_like(x)

    def cute_lowrank_gate_supported(x: torch.Tensor, wd: torch.Tensor,
                                    rank: int) -> bool:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                and wd.dtype == torch.bfloat16):
            return False
        n = x.shape[-1]
        if n % _VEC != 0:
            return False
        vecs = n // _VEC
        cluster_n = _default_cluster_n()
        if vecs % cluster_n != 0:
            return False
        return rank in (8, 16, 32, 64)

    def warmup_cute_lowrank_gate(n: int, r: int, device) -> None:
        """Compile ahead of CUDA graph capture."""
        x = torch.zeros(1, n, dtype=torch.bfloat16, device=device)
        wd = torch.zeros(r, n, dtype=torch.bfloat16, device=device)
        wu = torch.zeros(r, n, dtype=torch.bfloat16, device=device)
        cute_lowrank_gate(x, wd, wu)

else:

    def cute_lowrank_gate_supported(x, wd, rank) -> bool:  # noqa: ARG001
        return False

"""Owned-output fused Add+RMSNorm with typed I/O.

Derived from FlashInfer fused_add_rmsnorm.py, copyright (c) 2025 FlashInfer
team, Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0).
Constructor and reduction are reused. The local launch adds owned Y/S pointers.
"""

import cutlass
from cutlass import Float32, Int64, cute
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import predicate_k, row_reduce_sum_multirow

from operators.deepseek_v32.norm._layout import check_layout


class LocalFusedRMSNorm(FusedAddRMSNormKernel):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        cluster_n = self.cluster_n
        self.kernel(mX, mR, mW, mY, mS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
            )
            sR = smem.allocate_tensor(
                mR.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
            )
        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        cX = cute.local_tile(cute.make_identity_tensor(mX.shape), tiler_mn, (bidx, cluster_y))
        cR = cute.local_tile(cute.make_identity_tensor(mR.shape), tiler_mn, (bidx, cluster_y))
        cY = cute.local_tile(cute.make_identity_tensor(mY.shape), tiler_mn, (bidx, cluster_y))
        cS = cute.local_tile(cute.make_identity_tensor(mS.shape), tiler_mn, (bidx, cluster_y))
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        gS = cute.local_tile(mS, tiler_mn, (bidx, cluster_y))
        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=self.vec_size * mX.element_type.width,
        )
        copy_r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mR.element_type,
            num_bits_per_copy=self.vec_size * mR.element_type.width,
        )
        copy_w = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=self.vec_size * mW.element_type.width,
        )
        copy_y = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=self.vec_size * mY.element_type.width,
        )
        copy_s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mS.element_type,
            num_bits_per_copy=self.vec_size * mS.element_type.width,
        )
        if cutlass.const_expr(self.use_async_copy):
            async_x = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=self.vec_size * mX.element_type.width,
            )
            async_r = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mR.element_type,
                num_bits_per_copy=self.vec_size * mR.element_type.width,
            )
            tiled_x = cute.make_tiled_copy(async_x, tv_layout, tiler_mn)
            tiled_r = cute.make_tiled_copy(async_r, tv_layout, tiler_mn)
            canonical_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=self.copy_bits
            )
        else:
            tiled_x = cute.make_tiled_copy(copy_x, tv_layout, tiler_mn)
            tiled_r = cute.make_tiled_copy(copy_r, tv_layout, tiler_mn)
            canonical_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=self.copy_bits
            )
        tiled_w = cute.make_tiled_copy(copy_w, tv_layout, tiler_mn)
        tiled_y = cute.make_tiled_copy(copy_y, tv_layout, tiler_mn)
        tiled_s = cute.make_tiled_copy(copy_s, tv_layout, tiler_mn)
        canonical = cute.make_tiled_copy(canonical_atom, tv_layout, tiler_mn)
        check_layout("input_source_tv", tiled_x.layout_src_tv_tiled, canonical.layout_src_tv_tiled)
        check_layout(
            "residual_source_tv", tiled_r.layout_src_tv_tiled, canonical.layout_src_tv_tiled
        )
        check_layout("weight_source_tv", tiled_w.layout_src_tv_tiled, canonical.layout_src_tv_tiled)
        check_layout(
            "output_destination_tv", tiled_y.layout_dst_tv_tiled, canonical.layout_dst_tv_tiled
        )
        check_layout(
            "saved_destination_tv", tiled_s.layout_dst_tv_tiled, canonical.layout_dst_tv_tiled
        )

        tx, tr, tw, ty, ts = (
            tiled_x.get_slice(tidx),
            tiled_r.get_slice(tidx),
            tiled_w.get_slice(tidx),
            tiled_y.get_slice(tidx),
            tiled_s.get_slice(tidx),
        )
        canonical_coords = canonical.get_slice(tidx).partition_S(cX)
        canonical_fragment = cute.make_fragment_like(canonical_coords, Float32)
        tXgX = tx.partition_S(gX)
        tXcX = tx.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        tRgR = tr.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = tx.partition_D(sX)
            tRsR = tr.partition_D(sR)
        tWgW = tw.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = tx.retile(tWrW)
        tYgY, tSgS = ty.partition_D(gY), ts.partition_D(gS)
        tYrY, tSrS = cute.make_fragment_like(tYgY), cute.make_fragment_like(tSgS)
        check_layout("input_register", tXrX.layout, canonical_fragment.layout)
        check_layout("residual_register", tRrR.layout, canonical_fragment.layout)
        check_layout("weight_register", tXrW.layout, canonical_fragment.layout)
        check_layout("output_register", tYrY.layout, canonical_fragment.layout)
        check_layout("saved_register", tSrS.layout, canonical_fragment.layout)
        check_layout("input_coordinates", tXcX.layout, canonical_coords.layout)
        check_layout("residual_coordinates", tr.partition_S(cR).layout, canonical_coords.layout)

        tXpX = predicate_k(tXcX, limit=H)
        tRpR = predicate_k(tr.partition_S(cR), limit=H)
        tWpW = predicate_k(tw.partition_S(cX), limit=H)
        tYpY = predicate_k(ty.partition_D(cY), limit=H)
        tSpS = predicate_k(ts.partition_D(cS), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(async_x, tXgX, tXsX, pred=tXpX)
                cute.copy(async_r, tRgR, tRsR, pred=tRpR)
            cute.arch.cp_async_commit_group()
            cute.copy(copy_w, tWgW, tWrW, pred=tWpW)
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_x, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_r, tRgR, tRrR, pred=tRpR)
            cute.copy(copy_w, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in
        # Store a separate rounded result; the live FP32 h remains the sole
        # operand of squaring/reduction/normalization, exactly as in the vendor.
        tSrS.store(h.to(mS.element_type))
        if row_in_bounds:
            cute.copy(copy_s, tSrS, tSgS, pred=tSpS)
        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        tYrY.store(y.to(mY.element_type))
        if row_in_bounds:
            cute.copy(copy_y, tYrY, tYgY, pred=tYpY)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()
